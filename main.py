# main.py

import sys
import os
from layers.input_layer import InputParser
from layers.llm_client import LLMClient
from layers.pipeline import ChunkAnalyzePipeline
from layers.merge_layer import MergeLayer
from layers.final_memory import FinalMemoryLayer
from layers.prompt_generator import PromptGeneratorLayer


def main():
    chat_file_path = sys.argv[1] if len(sys.argv) > 1 else "chat.txt"

    print("=" * 55)
    print("        CHAT MEMORY SYSTEM — MVP")
    print("=" * 55)
    print(f"\n[Girdi] Dosya: {chat_file_path}")

    if not os.path.exists(chat_file_path):
        print(f"\n[HATA] '{chat_file_path}' bulunamadı!")
        print("Kullanım: python main.py <dosya.txt>")
        sys.exit(1)

    with open(chat_file_path, "r", encoding="utf-8") as f:
        raw_chat_history = f.read()

    print(f"[Girdi] ✓ {len(raw_chat_history)} karakter okundu.")

    # ----------------------------------------------------------------
    # KATMAN 1: Input Layer
    # ----------------------------------------------------------------
    print("\n[Katman 1] Girdi ayrıştırılıyor...")
    parser       = InputParser()
    cleaned_chat = parser.parse_raw_text(raw_chat_history)

    if not cleaned_chat:
        print("\n[HATA] Hiç mesaj ayrıştırılamadı!")
        sys.exit(1)

    print(f"[Katman 1] ✓ {len(cleaned_chat)} mesaj ayrıştırıldı.")
    for i, msg in enumerate(cleaned_chat):
        role    = "User" if msg["role"] == "user" else "AI  "
        preview = msg["text"][:60] + "..." if len(msg["text"]) > 60 else msg["text"]
        print(f"  [{i:02d}] {role} | {preview}")

    # ----------------------------------------------------------------
    # KATMAN 2 + 3: Chunking + Analyzing (paralel pipeline)
    # ----------------------------------------------------------------
    print("\n[Katman 2+3] Chunking ve Analyzing paralel başlıyor...")
    pipeline = ChunkAnalyzePipeline(
        max_tokens_per_block=2000,
        overlap_messages=3,
        min_chunk_size=2,
    )
    analyses = pipeline.run(cleaned_chat)

    print(f"\n[Katman 2+3] ✓ {len(analyses)} chunk analiz edildi.")
    for a in analyses:
        print(f"\n  Chunk {a['chunk_number']}:")
        print(f"    Topic          : {a['topic']}")
        print(f"    Decisions      : {a['decisions']}")
        print(f"    Open Questions : {a['open_questions']}")
        print(f"    Progress       : {a['progress']}")
        print(f"    Context        : {a['context']}")

    # ----------------------------------------------------------------
    # KATMAN 4: Merge Layer
    # ----------------------------------------------------------------
    print("\n[Katman 4] Chunk analizleri birleştiriliyor...")
    llm    = LLMClient()
    merger = MergeLayer(llm_client=llm)
    merged = merger.merge(analyses)

    print("\n[Katman 4] ✓ Birleştirildi.")
    print(f"  Topic          : {merged.get('topic', '')}")
    print(f"  Decisions      : {merged.get('decisions', [])}")
    print(f"  Open Questions : {merged.get('open_questions', [])}")
    print(f"  Progress       : {merged.get('progress', '')}")
    print(f"  Context        : {merged.get('context', '')}")

    # ----------------------------------------------------------------
    # KATMAN 5: Final Memory
    # ----------------------------------------------------------------
    print("\n[Katman 5] Final hafıza oluşturuluyor...")
    final_memory_layer = FinalMemoryLayer(llm_client=llm)
    memory = final_memory_layer.generate(merged)

    print("\n[Katman 5] ✓ Hafıza oluşturuldu.")
    print(f"\n{memory}")

    # ----------------------------------------------------------------
    # KATMAN 6: Prompt Generator
    # ----------------------------------------------------------------
    print("\n[Katman 6] Context prompt oluşturuluyor...")
    prompt_generator = PromptGeneratorLayer(llm_client=llm)
    final_prompt     = prompt_generator.generate(memory)

    print("\n" + "=" * 55)
    print("             FINAL PROMPT")
    print("=" * 55)
    print(f"\n{final_prompt}")
    print("\n" + "=" * 55)
    print("Pipeline tamamlandı — tüm katmanlar aktif")
    print("=" * 55)


if __name__ == "__main__":
    main()