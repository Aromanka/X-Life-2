"""
Image model configuration file
Used to configure CGM prediction models that support image input
"""

# Available language model configurations
LANGUAGE_MODELS = {
    'bert': {
        'name': 'bert-base-chinese',
        'hidden_size': 768,
        'description': 'BERT base Chinese model, suitable for general Chinese text processing'
    },
    # The manuscript uses the generative Qwen3-8B encoder for exercise and
    # physiological text features.  Keep the older embedding entry as an
    # explicit legacy option for backwards-compatible checkpoints.
    'qwen3-8b': {
        'name': 'Qwen/Qwen3-8B',
        'hidden_size': 4096,
        'description': 'Qwen3 8B language model used by the manuscript'
    },
    'qwen-7b': {
        'name': 'Qwen/Qwen-7B',
        'hidden_size': 4096,
        'description': 'Legacy Qwen language model configuration'
    },
    'qwen3-embedding-8b': {
        'name': 'Qwen/Qwen3-Embedding-8B',
        'hidden_size': 4096,
        'description': 'Legacy Qwen3 embedding model for older checkpoints'
    },
    'qwen3-embedding-0.6b': {
        'name': 'Qwen/Qwen3-Embedding-0.6B',
        'hidden_size': 1024,
        'description': 'Lightweight legacy embedding model for local smoke tests'
    }
}

# Available vision-language model configurations
VISION_MODELS = {
    'qwen2.5-vl-72b-instruct': {
        'name': 'Qwen/Qwen2.5-VL-72B-Instruct',
        'hidden_size': 8192,
        'description': 'Qwen2.5-VL-72B-Instruct diet image and text encoder from the manuscript'
    },
    'qwen3-vl-embedding-8b': {
        'name': 'Qwen/Qwen3-VL-Embedding-8B',
        'hidden_size': 1024,
        'description': 'Qwen3-VL 8B embedding model, specialized for image-text embedding'
    },
    'cnn-backup': {
        'name': 'CNN Backup',
        'hidden_size': 512,
        'description': 'Simple CNN encoder, backup solution for vision models'
    }
}

# Default model configurations
DEFAULT_TEXT_MODEL = 'qwen3-8b'  # Manuscript exercise/physiology text encoder
DEFAULT_VISION_MODEL = 'qwen2.5-vl-72b-instruct'  # Manuscript diet VLM

# Model parameter configurations
MODEL_CONFIGS = {
    'bert': {
        'text_hidden_dim': 256,
        'cgm_d_model': 128,
        'cgm_nhead': 4,
        'cgm_num_layers': 2,
        'cgm_dim_feedforward': 512,
        'dropout': 0.1
    },
    'qwen-7b': {
        'text_hidden_dim': 512,
        'cgm_d_model': 256,
        'cgm_nhead': 8,
        'cgm_num_layers': 4,
        'cgm_dim_feedforward': 1024,
        'dropout': 0.1
    },
    'qwen3-8b': {
        'text_hidden_dim': 256,
        'cgm_d_model': 128,
        'cgm_nhead': 4,
        'cgm_num_layers': 2,
        'cgm_dim_feedforward': 512,
        'dropout': 0.1
    },
    'qwen3-embedding-8b': {
        'text_hidden_dim': 256,
        'cgm_d_model': 128,
        'cgm_nhead': 4,
        'cgm_num_layers': 2,
        'cgm_dim_feedforward': 512,
        'dropout': 0.1
    },
    # Backwards-compatible alias used by the small CNN-only demonstration.
    'qwen3-embedding-0.6b': {
        'text_hidden_dim': 256,
        'cgm_d_model': 128,
        'cgm_nhead': 4,
        'cgm_num_layers': 2,
        'cgm_dim_feedforward': 512,
        'dropout': 0.1
    }
}

# Training configurations
TRAINING_CONFIGS = {
    'default': {
        'model_key': DEFAULT_TEXT_MODEL,
        'vision_model_key': DEFAULT_VISION_MODEL,
        'model_name': 'CGM Prediction with Image Features',
        'batch_size': 1,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'early_stopping_patience': 10,
        'lr_scheduler_patience': 5,
        'weight_decay': 1e-5,
        'gradient_clip_norm': 1.0,
        'log_interval': 10,
        'eval_interval': 1,
        'save_interval': 5,
        'use_image_features': True,
        'image_size': (224, 224),
        'gradient_accumulation_steps': 4,
        'training_phase': 'world_model',
        'freeze_action_encoders': True,
        'preference_beta': 0.1,
        'split_seeds': [42, 43, 44, 45, 46],
        'split_index': 0
    },
    'debug': {
        'model_key': DEFAULT_TEXT_MODEL,
        'vision_model_key': DEFAULT_VISION_MODEL,
        'model_name': 'CGM Prediction with Image Features (Debug)',
        'batch_size': 1,
        'learning_rate': 1e-4,
        'num_epochs': 2,
        'early_stopping_patience': 5,
        'lr_scheduler_patience': 2,
        'weight_decay': 1e-5,
        'gradient_clip_norm': 1.0,
        'log_interval': 1,
        'eval_interval': 1,
        'save_interval': 1,
        'use_image_features': True,
        'image_size': (224, 224),
        'gradient_accumulation_steps': 1,
        'training_phase': 'world_model',
        'freeze_action_encoders': True,
        'preference_beta': 0.1,
        'split_seeds': [42, 43, 44, 45, 46],
        'split_index': 0
    },
    'production': {
        'model_key': DEFAULT_TEXT_MODEL,
        'vision_model_key': DEFAULT_VISION_MODEL,
        'model_name': 'CGM Prediction with Image Features (Production)',
        'batch_size': 8,
        'learning_rate': 5e-5,
        'num_epochs': 100,
        'early_stopping_patience': 20,
        'lr_scheduler_patience': 10,
        'weight_decay': 1e-5,
        'gradient_clip_norm': 1.0,
        'log_interval': 20,
        'eval_interval': 1,
        'save_interval': 10,
        'use_image_features': True,
        'image_size': (224, 224),
        'gradient_accumulation_steps': 4,
        'training_phase': 'world_model',
        'freeze_action_encoders': True,
        'preference_beta': 0.1,
        'split_seeds': [42, 43, 44, 45, 46],
        'split_index': 0
    },
    'small_model': {
        'model_key': 'qwen3-embedding-0.6b',
        'vision_model_key': 'cnn-backup',
        'model_name': 'CGM Prediction with Small Models',
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 30,
        'early_stopping_patience': 8,
        'lr_scheduler_patience': 4,
        'weight_decay': 1e-5,
        'gradient_clip_norm': 1.0,
        'log_interval': 10,
        'eval_interval': 1,
        'save_interval': 5,
        'use_image_features': True,
        'image_size': (224, 224),
        'gradient_accumulation_steps': 4,
        'training_phase': 'encoder_pretraining',
        'freeze_action_encoders': False,
        'preference_beta': 0.1,
        'split_seeds': [42, 43, 44, 45, 46],
        'split_index': 0
    },
    'cnn_only': {
        'model_key': 'qwen3-embedding-0.6b',
        'vision_model_key': 'cnn-backup',
        'model_name': 'CGM Prediction with CNN Only',
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'early_stopping_patience': 10,
        'lr_scheduler_patience': 5,
        'weight_decay': 1e-5,
        'gradient_clip_norm': 1.0,
        'log_interval': 10,
        'eval_interval': 1,
        'save_interval': 5,
        'use_image_features': True,
        'image_size': (224, 224),
        'gradient_accumulation_steps': 4,
        'training_phase': 'world_model',
        'freeze_action_encoders': True,
        'preference_beta': 0.1,
        'split_seeds': [42, 43, 44, 45, 46],
        'split_index': 0
    }
}

def get_model_config(model_key=None):
    """Get text model configuration"""
    if model_key is None:
        model_key = DEFAULT_TEXT_MODEL
    
    if model_key not in LANGUAGE_MODELS:
        raise ValueError(f"Unknown text model: {model_key}, available models: {list(LANGUAGE_MODELS.keys())}")
    
    config = LANGUAGE_MODELS[model_key].copy()
    # A model entry must always have a matching architecture entry.  Keeping
    # this explicit avoids silently falling back to a different model size.
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Missing architecture configuration for text model: {model_key}")
    config.update(MODEL_CONFIGS[model_key])
    return config

def get_vision_model_config(model_key=None):
    """Get vision model configuration"""
    if model_key is None:
        model_key = DEFAULT_VISION_MODEL
    
    if model_key not in VISION_MODELS:
        raise ValueError(f"Unknown vision model: {model_key}, available models: {list(VISION_MODELS.keys())}")
    
    return VISION_MODELS[model_key].copy()

def get_training_config(config_name='default'):
    """Get training configuration"""
    if config_name not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown training configuration: {config_name}, available configurations: {list(TRAINING_CONFIGS.keys())}")
    
    return TRAINING_CONFIGS[config_name].copy()

def list_available_models():
    """List all available models"""
    print("Available text models:")
    print("-" * 60)
    for key, model in LANGUAGE_MODELS.items():
        print(f"{key:20} - {model['description']}")
    
    print("\nAvailable vision models:")
    print("-" * 60)
    for key, model in VISION_MODELS.items():
        print(f"{key:20} - {model['description']}")

def list_training_configs():
    """List all training configurations"""
    print("Available training configurations:")
    print("-" * 60)
    for key, config in TRAINING_CONFIGS.items():
        print(f"{key:15} - {config['model_name']}")
        print(f"{'':15}   Text model: {config['model_key']}")
        print(f"{'':15}   Vision model: {config.get('vision_model_key', 'None')}")
        print(f"{'':15}   Batch size: {config['batch_size']}")
        print(f"{'':15}   Learning rate: {config['learning_rate']}")
        print(f"{'':15}   Use images: {config['use_image_features']}")
        print()

def create_complete_config(config_name='default', **kwargs):
    """Create complete configuration including text model, vision model, and training parameters"""
    # Get base training configuration
    config = get_training_config(config_name)
    
    # Get text model configuration
    text_model_config = get_model_config(config['model_key'])
    
    # Get vision model configuration (if using images)
    if config['use_image_features'] and config['vision_model_key']:
        vision_model_config = get_vision_model_config(config['vision_model_key'])
    else:
        vision_model_config = None
    
    # Merge configurations
    complete_config = config.copy()
    complete_config.update(text_model_config)
    
    if vision_model_config:
        complete_config['vision_model_name'] = vision_model_config['name']
        complete_config['vision_hidden_size'] = vision_model_config['hidden_size']
    
    # Apply user-provided parameter overrides
    complete_config.update(kwargs)
    
    return complete_config
