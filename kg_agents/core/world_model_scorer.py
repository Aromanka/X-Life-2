"""Local checkpoint adapter for prescription ranking."""
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import sys
from threading import Lock

from config_loader import get_config


ROOT = Path(__file__).resolve().parents[2]


def _path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


class WorldModelScorer:
    def __init__(self, settings=None):
        self.settings = dict(settings if settings is not None else get_config().get('world_model', {}))
        self.model = None
        self.lock = Lock()

    def bind(self, plan_type, context=None):
        if plan_type not in {'diet', 'exercise'}:
            raise ValueError('Unknown prescription type.')
        return partial(self.score, plan_type=plan_type, context=deepcopy(context))

    def _load(self):
        if self.model is not None:
            return
        import torch
        directory = str(ROOT / 'world_model+ranking_module')
        if directory not in sys.path:
            sys.path.insert(0, directory)
        from model import CGMPredictionModelWithImage
        from dataset import get_tokenizer

        checkpoint_path = self.settings.get('checkpoint_path')
        if not checkpoint_path or not _path(checkpoint_path).is_file():
            raise ValueError('Set world_model.checkpoint_path to a local aligned checkpoint.')
        checkpoint = torch.load(_path(checkpoint_path), map_location='cpu', weights_only=True)
        if checkpoint.get('architecture_version') != 2 or checkpoint.get('test_backend', False):
            raise ValueError('Invalid world-model ranking checkpoint.')
        if checkpoint.get('phase') in {'encoder_pretraining', 'world_model'}:
            raise ValueError('Prescription ranking requires a preference-aligned checkpoint.')
        config = dict(checkpoint.get('config', {}))
        if self.settings.get('config_path'):
            config = json.loads(_path(self.settings['config_path']).read_text(encoding='utf-8'))
        model_config = dict(checkpoint['model_config'], include_training_heads=False)
        for key in ('qwen_model_path', 'vision_model_path'):
            value = config.get('model', {}).get(key) or model_config.get(key)
            if value:
                model_config[key] = str(_path(value))
        device = self.settings.get('device', 'auto')
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        model = CGMPredictionModelWithImage(**model_config)
        state = {key: value for key, value in checkpoint['model_state_dict'].items()
                 if not key.startswith(model.training_only_prefixes)}
        model.load_state_dict(state, strict=True)
        model.to(self.device).eval().requires_grad_(False)
        self.tokenizer = get_tokenizer(model_path=model_config.get('qwen_model_path'))
        self.data_config = dict(config.get('data', {}))
        if self.data_config.get('image_dir'):
            self.data_config['image_dir'] = str(_path(self.data_config['image_dir']))
        self.annotation_dim = model_config.get('annotation_dim', 4096)
        self.participant_id = checkpoint.get('participant_id')
        self.model = model

    @staticmethod
    def _diet(plan):
        if isinstance(plan.get('food_data'), dict):
            return deepcopy(plan['food_data'])
        fields = ('food', 'food_name', 'name', 'portion', 'weight_g', 'nutrition',
                  'nutrients', 'ingredients', 'preparation', 'img_names')
        items = [{key: item[key] for key in fields if key in item} for item in plan.get('items', [])]
        for original, item in zip(plan.get('items', []), items):
            if original.get('calories') is not None:
                item['nutrition'] = dict(item.get('nutrition') or {}, calories=original['calories'])
        data = {key: plan[key] for key in ('description', 'ingredients', 'preparation', 'nutrition',
                                         'nutrients', 'carbohydrate_g', 'protein_g', 'fat_g', 'fiber_g') if key in plan}
        if not data.get('description'):
            data['description'] = '; '.join(f"{item.get('food', item.get('food_name', item.get('name', '')))} {item.get('portion', '')}" for item in items)
        data['food_items'] = items
        if plan.get('total_calories') is not None:
            data['nutrition'] = dict(data.get('nutrition') or {}, calories=plan['total_calories'])
        return {'meal_type': plan.get('meal_type', ''), 'data': data}

    @staticmethod
    def _exercise(plan):
        def value(item, key):
            result = item.get(key)
            return getattr(result, 'value', result)
        if isinstance(plan.get('exercise_data'), dict):
            data = deepcopy(plan['exercise_data'])
            for key in ('meal_timing', 'start_minutes_after_meal'):
                timing = value(data, key)
                if timing is None:
                    timing = value(plan, key)
                if timing is not None:
                    data[key] = timing
            return data
        sessions = plan.get('sessions', {})
        sessions = list(sessions.values()) if isinstance(sessions, dict) else sessions
        exercises = [item for session in sessions for item in session.get('exercises', [])]
        return {
            'type': '; '.join(f"{item.get('name', '')} ({value(item, 'exercise_type') or ''})" for item in exercises),
            'duration': plan.get('total_duration_minutes', sum(item.get('duration_minutes', 0) for item in exercises)),
            'intensity': [value(item, 'intensity') for item in exercises],
            'MET': [item.get('MET', item.get('met')) for item in exercises if item.get('MET', item.get('met')) is not None] or None,
            'calories': plan.get('total_calories_burned', 0),
            'meal_timing': value(plan, 'meal_timing'),
            'start_minutes_after_meal': plan.get('start_minutes_after_meal'),
        }

    def score(self, plan, user_metadata, environment=None, *, plan_type, context=None):
        context = context if context is not None else user_metadata.get('world_model_context')
        if not isinstance(context, dict):
            raise ValueError('Supply world_model_context with the current physiological event.')
        record = deepcopy(context)
        record.setdefault('subject_id', user_metadata.get('subject_id'))
        record.setdefault('metadata', deepcopy(user_metadata))
        metadata = record['metadata']
        if metadata.get('height_cm') is not None:
            metadata.setdefault('height', float(metadata['height_cm']) / 100)
        if metadata.get('weight_kg') is not None:
            metadata.setdefault('weight', metadata['weight_kg'])
        record.setdefault('cgm_pre', record.get('cgm_preprandial', record.get('cgm_pre_exercise', [])))
        if not record['cgm_pre']:
            raise ValueError('world_model_context requires measured pre-event CGM.')
        record['cgm_post'], record['training_annotations'], record['training_targets'] = [], {}, {}
        with self.lock:
            self._load()
            from train_data import normalize_event, event_dataset, collate_events
            from training import model_inputs, to_device
            import torch
            record = normalize_event(record)
            if self.participant_id and str(record['subject_id']) != str(self.participant_id):
                raise ValueError('The ranking checkpoint belongs to another participant.')
            record['food_data' if plan_type == 'diet' else 'exercise_data'] = (
                self._diet(plan) if plan_type == 'diet' else self._exercise(plan))
            dataset = event_dataset([record], self.tokenizer, self.data_config, self.annotation_dim, require_cgm=False)
            batch = to_device(collate_events([dataset[0]]), self.device)
            if not batch['cgm_preprandial_mask'].any():
                raise ValueError('No valid CGM measurements remain in the decision window.')
            with torch.inference_mode():
                result = self.model(**model_inputs(batch), return_details=True, decode_glucose=False)
            score = result['ranking_score'].reshape(-1)[0]
            if not torch.isfinite(score):
                raise ValueError('The ranking model returned a nonfinite score.')
            return float(score)
