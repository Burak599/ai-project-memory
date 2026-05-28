# layers/llm_client.py

import time
from groq import Groq, RateLimitError, InternalServerError
import config


class LLMClient:
    """
    Groq API için ortak arayüz.
    Her katman kendi model adını geçirerek kullanır.
    Rate limit hatalarında otomatik retry + exponential backoff uygular.
    413 (Groq'ta TPM rate limit) ve 429 her ikisi de yakalanır.
    """

    MAX_RETRIES = 10000000
    BASE_WAIT   = 1
    TPM_BUDGET_ESTIMATE = 5600
    DEFAULT_MAX_TOKENS = 1200
    MIN_MAX_TOKENS = 200

    def __init__(self):
        self._client = Groq(api_key=config.get_groq_key())
        print(f"[LLM Client] Groq aktif.")

    def chat(self, model: str, system_prompt: str, user_message: str) -> str:
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                dynamic_max_tokens = self._compute_max_tokens(system_prompt, user_message)
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=0,
                    max_tokens=dynamic_max_tokens,
                )
                return response.choices[0].message.content.strip()

            except RateLimitError as e:
                last_error = e
                wait = self.BASE_WAIT ** attempt
                print(f"[LLM Client] Rate limit (429) — {wait}s bekleniyor... (deneme {attempt + 1}/{self.MAX_RETRIES})")
                time.sleep(wait)

            except InternalServerError as e:
                last_error = e
                wait = self.BASE_WAIT ** attempt
                print(f"[LLM Client] Sunucu hatası — {wait}s bekleniyor... (deneme {attempt + 1}/{self.MAX_RETRIES})")
                time.sleep(wait)

            except Exception as e:
                err = str(e)
                # Groq'ta 413 = TPM rate limit, retry'a alınmalı
                raise e

        raise Exception(f"[LLM Client] {self.MAX_RETRIES} denemede başarısız. Son hata: {last_error}")

    def _compute_max_tokens(self, system_prompt: str, user_message: str) -> int:
        """
        Keep total requested tokens under common on_demand TPM caps.
        Approximation is intentionally conservative.
        """
        prompt_chars = len(system_prompt) + len(user_message)
        prompt_tokens_est = max(1, prompt_chars // 4)
        remaining = self.TPM_BUDGET_ESTIMATE - prompt_tokens_est
        capped = min(self.DEFAULT_MAX_TOKENS, remaining)
        return max(self.MIN_MAX_TOKENS, capped)
