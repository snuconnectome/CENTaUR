"""
BaselineModelManager: Unified interface for loading and running all baseline models

TDD Implementation: Designed to pass all tests in test_baseline_manager.py
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from typing import Optional, Dict
from pathlib import Path


class BaselineModelManager:
    """
    Unified interface for loading and running all baseline models

    Supports:
    - Ko-CENTaUR (EXAONE + Psych-101 fine-tuning)
    - EXAONE-base (pre-training only)
    - Llama-Centaur-70B (Marcel Binz's CENTaUR model)
    - Llama-3.1-70B-Instruct (base model)
    - SOLAR-10.7B (Korean LLM)
    """

    # Model registry with descriptions
    MODELS = {
        "ko-centaur": "Ko-CENTaUR (EXAONE + Psych-101 fine-tuning)",
        "exaone-base": "EXAONE-3.0-7.8B-Instruct (base model)",
        "llama-centaur": "Llama-3.1-Centaur-70B (Marcel Binz CENTaUR)",
        "llama-base": "Llama-3.1-70B-Instruct (base model)",
        "solar": "SOLAR-10.7B-Instruct (Korean LLM)"
    }

    # GPU requirements per model
    GPU_REQUIREMENTS = {
        "ko-centaur": 1,
        "exaone-base": 1,
        "llama-centaur": 3,
        "llama-base": 3,
        "solar": 1
    }

    # Checkpoint paths
    KO_CENTAUR_CHECKPOINT = "/scratch/connectome/connectome1/ko-centaur/models/ko-centaur-full/checkpoint-22536"
    BASELINES_DIR = "/scratch/connectome/connectome1/ko-centaur/models/baselines"

    def __init__(self, model_type: str, device_map: Optional[Dict] = None):
        """
        Initialize baseline model manager

        Args:
            model_type: One of MODELS keys
            device_map: Optional device mapping for multi-GPU models
        """
        if model_type not in self.MODELS:
            raise ValueError(
                f"Invalid model_type '{model_type}'. "
                f"Must be one of: {list(self.MODELS.keys())}"
            )

        self.model_type = model_type
        self.device_map = device_map
        self.model, self.tokenizer = self._load_model()
        self.hidden_size = self.model.config.hidden_size

    def _load_model(self):
        """Load model and tokenizer based on model_type"""

        if self.model_type == "ko-centaur":
            return self._load_ko_centaur()

        elif self.model_type == "exaone-base":
            return self._load_exaone_base()

        elif self.model_type == "llama-centaur":
            return self._load_llama_centaur()

        elif self.model_type == "llama-base":
            return self._load_llama_base()

        elif self.model_type == "solar":
            return self._load_solar()

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _load_ko_centaur(self):
        """Load Ko-CENTaUR (EXAONE + Psych-101 LoRA weights)"""
        print(f"Loading Ko-CENTaUR from {self.KO_CENTAUR_CHECKPOINT}...")

        # Load base EXAONE model
        base_model = AutoModelForCausalLM.from_pretrained(
            "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True
        )

        # Load LoRA weights
        model = PeftModel.from_pretrained(
            base_model,
            self.KO_CENTAUR_CHECKPOINT
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
        )

        print(f"✓ Ko-CENTaUR loaded (hidden_size={base_model.config.hidden_size})")
        return model, tokenizer

    def _load_exaone_base(self):
        """Load EXAONE base model (no fine-tuning)"""
        print("Loading EXAONE-base...")

        model = AutoModelForCausalLM.from_pretrained(
            "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
        )

        print(f"✓ EXAONE-base loaded (hidden_size={model.config.hidden_size})")
        return model, tokenizer

    def _load_llama_centaur(self):
        """Load Llama-3.1-Centaur-70B (Marcel Binz's CENTaUR)"""
        model_path = Path(self.BASELINES_DIR) / "llama-centaur-70b"
        print(f"Loading Llama-Centaur-70B from {model_path}...")

        # Default device map for 3 GPUs if not specified
        if self.device_map is None:
            self.device_map = "auto"
            max_memory = {3: "22GB", 4: "22GB", 5: "22GB"}
        else:
            max_memory = self.device_map

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            load_in_4bit=True,
            device_map=self.device_map,
            max_memory=max_memory,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-70B-Instruct"
        )

        print(f"✓ Llama-Centaur-70B loaded (hidden_size={model.config.hidden_size})")
        return model, tokenizer

    def _load_llama_base(self):
        """Load Llama-3.1-70B-Instruct base model"""
        print("Loading Llama-3.1-70B-Instruct...")

        # Default device map for 3 GPUs
        if self.device_map is None:
            self.device_map = "auto"
            max_memory = {0: "22GB", 1: "22GB", 2: "22GB"}
        else:
            max_memory = self.device_map

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-70B-Instruct",
            load_in_4bit=True,
            device_map=self.device_map,
            max_memory=max_memory,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-70B-Instruct"
        )

        print(f"✓ Llama-70B-base loaded (hidden_size={model.config.hidden_size})")
        return model, tokenizer

    def _load_solar(self):
        """Load SOLAR-10.7B Korean LLM"""
        model_path = Path(self.BASELINES_DIR) / "solar-10.7b"
        print(f"Loading SOLAR-10.7B from {model_path}...")

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "upstage/SOLAR-10.7B-Instruct-v1.0"
        )

        print(f"✓ SOLAR-10.7B loaded (hidden_size={model.config.hidden_size})")
        return model, tokenizer

    def format_prompt(self, task_description: str, system_prompt: str = "당신은 전문 심리학자입니다.") -> str:
        """
        Format prompt according to model's chat template

        Args:
            task_description: User's task/question
            system_prompt: System instruction

        Returns:
            Formatted prompt string ready for model input
        """
        if self.model_type in ["ko-centaur", "exaone-base"]:
            # EXAONE chat template
            return f"<|system|>{system_prompt}<|user|>{task_description}<|assistant|>"

        elif self.model_type in ["llama-centaur", "llama-base"]:
            # Llama chat template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_description}
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        elif self.model_type == "solar":
            # SOLAR chat template
            return f"### System:\n{system_prompt}\n\n### User:\n{task_description}\n\n### Assistant:\n"

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def extract_features(self, prompt: str) -> torch.Tensor:
        """
        Extract last-layer hidden states (CENTaUR methodology)

        Args:
            prompt: Formatted prompt string

        Returns:
            Hidden state tensor of shape [batch_size, hidden_size]
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move to model device
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate with hidden states output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,  # CENTaUR methodology: single token
                temperature=0.0,    # Deterministic
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        # Extract last token's last layer hidden state
        # outputs.hidden_states is tuple of tuples: (generation_step, layer, tensor)
        last_generation_step = outputs.hidden_states[-1]  # Last generation step
        last_layer = last_generation_step[-1]  # Last layer

        # Get last token position: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        features = last_layer[:, -1, :].cpu()

        return features

    def __repr__(self):
        return f"BaselineModelManager(model_type='{self.model_type}', hidden_size={self.hidden_size})"


def list_available_models():
    """List all available models and their descriptions"""
    print("\n" + "="*60)
    print("Available Baseline Models")
    print("="*60)

    for model_type, description in BaselineModelManager.MODELS.items():
        gpu_req = BaselineModelManager.GPU_REQUIREMENTS[model_type]
        print(f"\n{model_type}:")
        print(f"  Description: {description}")
        print(f"  GPU Required: {gpu_req}")

    print("\n" + "="*60)


if __name__ == "__main__":
    # Show available models
    list_available_models()
