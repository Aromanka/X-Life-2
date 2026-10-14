import os
import torch
from typing import List, Dict, Optional, Any
from datetime import datetime
from config_loader import get_config
from core.llm.utils import parse_json_response

_local_model = None
_local_processor = None


def _get_local_model_path() -> Optional[str]:
    """Get local model path from config"""
    try:
        config = get_config()
        return config.get("local_model_path")
    except Exception:
        return None


_load_failed = False


def _load_local_model():
    global _local_model, _local_processor, _load_failed

    if _local_model is not None:
        return _local_model, _local_processor

    if _load_failed:
        raise RuntimeError("Local model previously failed to load, skipping retry")

    model_path = _get_local_model_path()
    if not model_path or not os.path.isdir(os.path.expanduser(model_path)):
        raise ValueError("local_model_path must be an existing local model directory")
    model_path = os.path.expanduser(model_path)

    print(f"[INFO] Loading local model from: {model_path}")

    try:
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
        config = get_config()
        multimodal = config.get("local_model_kind", "text") == "multimodal"
        model_class = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
        processor_class = AutoProcessor if multimodal else AutoTokenizer

        _local_model = model_class.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation=config.get("local_attn_implementation", "sdpa"),
            device_map="auto",
            local_files_only=True,
        )
        _local_processor = processor_class.from_pretrained(model_path, local_files_only=True)
        print(f"[INFO] Local model loaded successfully")
        return _local_model, _local_processor
    except Exception as e:
        print(f"[ERROR] Failed to load local model: {e}")
        _load_failed = True
        raise


def _get_log_path() -> str:
    """Get log path from config"""
    try:
        config = get_config()
        log_path = config.get("llm_log_path", "logs/llm.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        return log_path
    except Exception:
        return "logs/llm.log"


def _log_to_file(messages: List[Dict[str, str]], response: Any, duration_ms: float = 0):
    """Log LLM request to file"""
    try:
        from core.llm.utils import parse_messages_to_str, parse_response_to_str

        output_text = f">>> model <local> generated at <{datetime.now().isoformat()}>\n"
        output_text += f"- query:\n{parse_messages_to_str(messages)}\n"
        output_text += f"- response:\n{parse_response_to_str(response)}\n"
        output_text += f"- duration <{duration_ms}> ms\n\n"

        log_path = _get_log_path()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(output_text)
    except Exception as e:
        print(f"[WARN] Failed to write LLM log: {e}")


class LocalLLM:
    """Local LLM wrapper compatible with the LLMClient interface"""

    def __init__(self):
        self.model_name = "local"
        self._log_path = _get_log_path()

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ) -> str:
        """Send chat request, return text content"""
        start_time = datetime.now()

        # Convert messages to text format for local model
        text_content = self._messages_to_text(messages)

        model, processor = _load_local_model()

        # Apply chat template
        inputs = processor.apply_chat_template(
            text_content,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Generate
        max_new_tokens = max_tokens if max_tokens else 2048
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )

        # Trim and decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        content = output_text[0] if output_text else ""

        _log_to_file(messages, {"content": content}, duration_ms)
        return content

    def chat_with_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ) -> dict:
        """Send chat request, return parsed JSON"""
        import json
        import re

        start_time = datetime.now()

        text_content = self._messages_to_text(messages)
        model, processor = _load_local_model()

        inputs = processor.apply_chat_template(
            text_content,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        content = output_text[0] if output_text else ""
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Handle empty response
        if not content or not content.strip():
            print(f"[WARN] LLM returned empty response")
            _log_to_file(messages, {"content": "", "error": "empty response"}, duration_ms)
            return {}

        # Extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            content = json_match.group(1)

        # Parse JSON
        try:
            result = parse_json_response(content)
            _log_to_file(messages, result, duration_ms)
            return result
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON parsing failed: {e}")
            _log_to_file(messages, {"error": str(e)}, duration_ms)
            return {}

    def extract_keywords(self, question: str, max_count: int = 3) -> List[str]:
        """Extract keywords from question"""
        import json
        import re

        prompt = f"Extract {max_count} medical/health entity keywords from the following question, only return JSON list format:\nQuestion: {question}"
        messages = [{"role": "user", "content": prompt}]

        try:
            content = self.chat(messages, temperature=0.1)
            match = re.search(r'\[.*\]', content)
            if match:
                return json.loads(match.group())
            return []
        except Exception as e:
            print(f"Failed to extract keywords: {e}")
            return []

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert message dicts to format expected by local model"""
        # Qwen3-VL apply_chat_template expects content as list of blocks
        if get_config().get("local_model_kind", "text") != "multimodal":
            return messages
        converted = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                # Wrap string content in list format
                converted.append({
                    "role": msg.get("role", "user"),
                    "content": [{"type": "text", "text": content}]
                })
            else:
                converted.append(msg)
        return converted


# Global instance
_local_llm_instance: Optional[LocalLLM] = None


def get_local_llm() -> LocalLLM:
    """Get the global local LLM instance"""
    global _local_llm_instance
    if _local_llm_instance is None:
        _local_llm_instance = LocalLLM()
    return _local_llm_instance


def unload_local_model():
    """Unload local model from memory (for memory cleanup)"""
    global _local_model, _local_processor, _local_llm_instance
    if _local_model is not None:
        del _local_model
        _local_model = None
    if _local_processor is not None:
        del _local_processor
        _local_processor = None
    _local_llm_instance = None
    print("[INFO] Local model unloaded from memory")


def is_local_mode() -> bool:
    """Check if local model is configured (field must exist in config)"""
    try:
        config = get_config()
        # Only use local mode if 'local_model_path' field explicitly exists
        return bool(config.get("local_model_path"))
    except Exception:
        return False
