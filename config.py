# config.py

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Her katmanın kendi modeli
LAYER_2_MODEL = "llama-3.1-8b-instant"    # Chunking — hızlı, basit görev
LAYER_3_MODEL = "qwen/qwen3-32b"          # Chunk analizi — en kaliteli
LAYER_4_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"    # Merge — hızlı, basit görev
LAYER_5_MODEL = "openai/gpt-oss-120b"     # Final memory — en güçlü
LAYER_6_MODEL = "llama-3.3-70b-versatile" # Prompt üretimi — dengeli
CODE_ANALYZER_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"
CODE_RELATION_MODEL  = "llama-3.3-70b-versatile"
CODE_MERGE_MODEL     = "meta-llama/llama-4-scout-17b-16e-instruct"
CODE_MEMORY_MODEL    = "qwen/qwen3-32b"
COMBINED_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Detail extractors (chat_detail + code_detail) should use a fast/stable model.
DETAIL_MODEL = LAYER_4_MODEL
CODE_DETAIL_MODEL = LAYER_4_MODEL

def get_groq_key() -> str:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "\n[HATA] GROQ_API_KEY bulunamadı!\n"
            "Adımlar:\n"
            "  1. cp .env.example .env\n"
            "  2. .env dosyasını aç, GROQ_API_KEY'ini gir\n"
            "  3. Tekrar çalıştır\n"
        )
    return GROQ_API_KEY