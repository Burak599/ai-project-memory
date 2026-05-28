# code_layers/code_relation_layer.py

import json
import re
from typing import List, Dict
from layers.llm_client import LLMClient
import config

RELATION_SYSTEM_PROMPT = """You are a software architect. You will receive a list of JSON summaries, one per file in a project.
Your job is to analyze the relationships between these files and produce a high-level architecture map.

Return ONLY a JSON object with exactly these fields:

{
  "architecture": "one paragraph describing the overall project structure and how it works",
  "core_modules": ["list of the most critical files the project depends on"],
  "entry_points": ["list of files that are the main entry points (main.py, app.py, cli.py etc.)"],
  "relations": [
    "fileA.py → fileB.py: reason why A depends on B",
    "fileC.py → fileD.py, fileE.py: reason"
  ],
  "hubs": ["files that are imported by many others — central dependencies"]
}

Rules:
- "relations" should be human-readable strings, not nested objects.
- Focus on project-internal relationships only. Ignore stdlib and third-party libs.
- "hubs" are files that appear in many other files' dependencies — they are the backbone.
- Be concise. No fluff.
- Return ONLY the JSON object. No markdown, no explanation, no extra text."""


class CodeRelationLayer:
    """
    Tüm dosya özetlerini birlikte analiz eder ve
    dosyalar arası bağımlılık haritası çıkarır.

    Girdi : CodeAnalyzerLayer.analyze_all() çıktısı
    Çıktı : Bağımlılık haritası dict'i
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def map(self, analyses: List[Dict]) -> Dict:
        """
        Args:
            analyses: CodeAnalyzerLayer.analyze_all() çıktısı

        Returns:
            {
                "architecture": str,
                "core_modules": [str],
                "entry_points": [str],
                "relations":    [str],
                "hubs":         [str],
            }
        """
        if not analyses:
            return {}

        if len(analyses) == 1:
            return {
                "architecture": analyses[0].get("purpose", ""),
                "core_modules": [analyses[0].get("file", "")],
                "entry_points": [analyses[0].get("file", "")],
                "relations":    [],
                "hubs":         [],
            }

        print(f"[Katman 3] {len(analyses)} dosya özeti ilişki analizine gönderiliyor...")

        user_message = (
            f"Analyze the relationships between these {len(analyses)} files:\n\n"
            f"{json.dumps(analyses, indent=2)}"
        )

        try:
            response = self.llm.chat(
                config.CODE_RELATION_MODEL,
                RELATION_SYSTEM_PROMPT,
                user_message,
            )
            return self._parse_response(response)

        except Exception as e:
            print(f"[Katman 3] Hata: {e}")
            return {}

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _parse_response(self, response: str) -> Dict:
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()

        try:
            data = json.loads(cleaned)
            return {
                "architecture": data.get("architecture", ""),
                "core_modules": data.get("core_modules", []),
                "entry_points": data.get("entry_points", []),
                "relations":    data.get("relations", []),
                "hubs":         data.get("hubs", []),
            }
        except json.JSONDecodeError:
            print(f"[Katman 3] JSON parse hatası, ham yanıt: {response[:100]}")
            return {}