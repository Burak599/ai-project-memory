# code_layers/code_analyzer.py

import json
import re
from typing import List, Dict
from layers.llm_client import LLMClient
import config

CODE_ANALYZER_SYSTEM_PROMPT = """You are a code analyst. You will receive a single source code file.
Extract the most important information from it and return ONLY a JSON object with exactly these fields:

{
  "file": "relative path of the file",
  "purpose": "one sentence: what does this file do?",
  "classes": ["list of class names defined in this file"],
  "functions": ["list of function/method names defined in this file"],
  "dependencies": ["list of internal modules this file imports (not stdlib, not third-party)"],
  "notes": "any important logic, patterns, or decisions worth remembering (empty string if nothing special)"
}

Rules:
- Be concise. No fluff.
- "dependencies" should only include project-internal imports, not stdlib (os, re, json) or third-party (torch, groq).
- If a field has nothing relevant, use an empty list [] or empty string "".
- Return ONLY the JSON object. No markdown, no explanation, no extra text."""


class CodeAnalyzerLayer:
    """
    Her dosyayı tek tek LLM'e gönderir ve kısa JSON özeti çıkarır.

    Çıktı formatı:
    {
        "file": str,
        "purpose": str,
        "classes": [str, ...],
        "functions": [str, ...],
        "dependencies": [str, ...],
        "notes": str
    }
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def analyze_all(self, files: List[Dict]) -> List[Dict]:
        """
        Tüm dosyaları sırayla analiz eder.

        Args:
            files: CodeInputLayer.scan() çıktısı

        Returns:
            Her dosya için analiz sonucu dict listesi
        """
        results = []
        for f in files:
            print(f"[Katman 2]   [{f['index']:02d}/{len(files):02d}] {f['path']} analiz ediliyor...")
            result = self._analyze_file(f)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _analyze_file(self, file_info: Dict) -> Dict:
        """Tek bir dosyayı analiz eder."""
        user_message = (
            f"File: {file_info['path']}\n\n"
            f"```{file_info['extension'].lstrip('.')}\n"
            f"{file_info['content']}\n"
            f"```"
        )

        try:
            response = self.llm.chat(
                config.CODE_ANALYZER_MODEL,
                CODE_ANALYZER_SYSTEM_PROMPT,
                user_message,
            )
            return self._parse_response(response, file_info["path"])

        except Exception as e:
            print(f"[Katman 2] Hata ({file_info['path']}): {e}")
            return self._empty_result(file_info["path"])

    def _parse_response(self, response: str, file_path: str) -> Dict:
        """LLM yanıtından JSON parse eder."""
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()

        try:
            data = json.loads(cleaned)
            return {
                "file":         data.get("file", file_path),
                "purpose":      data.get("purpose", ""),
                "classes":      data.get("classes", []),
                "functions":    data.get("functions", []),
                "dependencies": data.get("dependencies", []),
                "notes":        data.get("notes", ""),
            }
        except json.JSONDecodeError:
            print(f"[Katman 2] JSON parse hatası ({file_path}), ham yanıt: {response[:100]}")
            return self._empty_result(file_path)

    def _empty_result(self, file_path: str) -> Dict:
        return {
            "file":         file_path,
            "purpose":      "",
            "classes":      [],
            "functions":    [],
            "dependencies": [],
            "notes":        "",
        }