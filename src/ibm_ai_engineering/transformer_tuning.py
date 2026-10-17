"""Generative AI Fine-Tuning and PEFT / LoRA Utilities.

Provides configuration helpers for Parameter-Efficient Fine-Tuning (PEFT),
Low-Rank Adaptation (LoRA), and Hugging Face Transformers quantization setups.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class LoRAConfigBuilder:
    """LoRA parameter configuration builder."""
    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_dict(self) -> Dict[str, Any]:
        """Export parameters as dictionary matching peft.LoraConfig."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type
        }


def format_instruction_prompt(instruction: str, input_text: Optional[str] = None, response: Optional[str] = None) -> str:
    """Standardize Alpaca-style instruction tuning format."""
    if input_text and input_text.strip():
        prompt = (
            f"Below is an instruction that describes a task, paired with an input that provides further context. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )
    if response:
        prompt += f"{response}"
    return prompt
