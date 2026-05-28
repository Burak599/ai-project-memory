# layers/prompt_generator.py

import re
from layers.llm_client import LLMClient
import config


PROMPT_GENERATOR_SYSTEM_PROMPT = """You are a prompt engineer. You will receive a plain text memory document.

Convert it into a context prompt for a new AI assistant. Do not omit, compress, or reword anything. Every sentence in the memory must appear in the output. Reformat only.

Write in second person. Return only the prompt text. Nothing else."""


class PromptGeneratorLayer:
    """
    Final hafızayı yeni bir AI'ya verilecek optimal context prompt'a dönüştürür.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(self, memory: str) -> str:
        """
        Args:
            memory: FinalMemoryLayer.generate() çıktısı

        Returns:
            Yeni AI'ya verilecek context prompt
        """
        if not memory:
            return ""

        user_message = f"Convert this memory into a context prompt for a new AI assistant:\n\n{memory}"

        try:
            response = self.llm.chat(
                config.LAYER_6_MODEL,
                PROMPT_GENERATOR_SYSTEM_PROMPT,
                user_message,
            )
            cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            return cleaned

        except Exception as e:
            print(f"[Katman 6] Hata: {e}")
            return ""