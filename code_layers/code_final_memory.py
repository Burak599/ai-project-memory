# code_layers/code_final_memory.py

import re
from typing import Dict
from layers.llm_client import LLMClient
import config

CODE_MEMORY_SYSTEM_PROMPT = """You are a senior software architect writing a project briefing document.
You will receive a unified JSON summary of a software project — its files, architecture, dependencies, and relationships.

Write a clear, concise plain text memory document that an AI assistant can read to immediately understand this codebase.

Structure it exactly like this (use these exact headers):

PROJECT: <project name>

PURPOSE:
One paragraph explaining what this project does and why it exists.

ARCHITECTURE:
One paragraph explaining how the project is structured. Mention the main layers or modules and how they interact.

ENTRY POINTS:
List the main entry point files and what each one does.

KEY FILES:
List the most important files (hubs and core modules), one line each: filename — what it does.

RELATIONSHIPS:
List the key dependencies between files, one line each.

Rules:
- Write in plain text only. No markdown, no bullet points, no JSON.
- Be concise but complete. A developer should understand the project in 30 seconds.
- Do NOT include debug scripts or test files as entry points.
- Do NOT repeat the same information in multiple sections.
- Return only the memory document. Nothing else."""


class CodeFinalMemoryLayer:
    """
    Birleşik JSON proje özetini düz metin hafıza belgesine çevirir.

    Girdi : CodeMergeLayer.merge() çıktısı
    Çıktı : Düz metin proje özeti (AI'a verilecek context)
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(self, merged: Dict) -> str:
        """
        Args:
            merged: CodeMergeLayer.merge() çıktısı

        Returns:
            Düz metin proje hafızası
        """
        if not merged:
            return ""

        print("[Katman 5] Final kod hafızası oluşturuluyor...")

        import json
        user_message = (
            f"Convert this project summary into a memory document:\n\n"
            f"{json.dumps(merged, indent=2)}"
        )

        try:
            response = self.llm.chat(
                config.CODE_MEMORY_MODEL,
                CODE_MEMORY_SYSTEM_PROMPT,
                user_message,
            )
            # <think> bloğunu temizle — birden fazla olabilir, nested olabilir
            cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            # Hâlâ açık <think> varsa (kapanmamış) → oradan sonrasını at
            if "<think>" in cleaned:
                cleaned = cleaned[:cleaned.index("<think>")]
            cleaned = cleaned.strip()
            return cleaned

        except Exception as e:
            print(f"[Katman 5] Hata: {e}")
            return ""