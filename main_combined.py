# main_combined.py
# Kullanım: python main_combined.py chat.txt /path/to/project

import sys
import os

# Chat pipeline
from layers.input_layer import InputParser
from layers.llm_client import LLMClient
from layers.chunking_layer import ChunkingLayer
from layers.chunk_analyzer import ChunkAnalyzer
from layers.merge_layer import MergeLayer
from layers.final_memory import FinalMemoryLayer
from layers.detail_keyword_layer import DetailKeywordExtractorLayer, DetailKeywordMergeLayer

# Kod pipeline
from code_layers.code_input_layer import CodeInputLayer
from code_layers.code_analyzer import CodeAnalyzerLayer
from code_layers.code_relation_layer import CodeRelationLayer
from code_layers.code_merge_layer import CodeMergeLayer
from code_layers.code_final_memory import CodeFinalMemoryLayer
from code_layers.code_detail_layer import CodeDetailExtractorLayer, CodeDetailMergeLayer

# Birleştirici
from layers.combined_memory import CombinedMemoryLayer


def run_chat_pipeline(chat_file_path: str, llm: LLMClient) -> tuple[str, str]:
    print("\n" + "=" * 55)
    print("  [CHAT PİPELİNE] Başlıyor...")
    print("=" * 55)

    if not os.path.exists(chat_file_path):
        print(f"[HATA] '{chat_file_path}' bulunamadı, chat pipeline atlanıyor.")
        return "", ""

    with open(chat_file_path, "r", encoding="utf-8") as f:
        raw_chat = f.read()
    print(f"[Chat | Katman 1] ✓ {len(raw_chat)} karakter okundu.")

    # Katman 1: Input Parser
    parser = InputParser()
    cleaned_chat = parser.parse_raw_text(raw_chat)
    if not cleaned_chat:
        print("[Chat | Katman 1] HATA: Hiç mesaj ayrıştırılamadı.")
        return "", ""
    print(f"[Chat | Katman 1] ✓ {len(cleaned_chat)} mesaj ayrıştırıldı.")

    # Shared chunking path: chunk once, reuse everywhere
    print("[Chat | Shared Chunking] Tek sefer chunk hazırlanıyor...")
    detail_chunker = ChunkingLayer(
        max_tokens_per_block=2000,
        overlap_messages=3,
        min_chunk_size=2,
    )
    detail_chunks = detail_chunker.chunk(cleaned_chat)
    detail_chunk_texts = detail_chunker.get_chunk_texts(detail_chunks)
    print(f"[Chat | Shared Chunking] ✓ {len(detail_chunk_texts)} chunk üretildi.")
    if not detail_chunk_texts:
        print("[Chat | DEBUG] UYARI: 0 chunk — chunking boş döndü, detail path çalışmaz.")
    else:
        total_chunk_chars = sum(len(t) for t in detail_chunk_texts)
        print(
            f"[Chat | DEBUG] Chunk boyutları: "
            f"min={min(len(t) for t in detail_chunk_texts)}, "
            f"max={max(len(t) for t in detail_chunk_texts)}, "
            f"toplam={total_chunk_chars} chars"
        )

    detail_extractor = DetailKeywordExtractorLayer(llm_client=llm, debug=True)
    detail_merge = DetailKeywordMergeLayer()
    extracted_details = detail_extractor.extract_all(detail_chunk_texts)
    merged_detail = detail_merge.merge(extracted_details, debug=True)
    detail_memory = merged_detail.get("memory_text", "")
    print(
        f"[Chat | Detail Path] ✓ {len(extracted_details)} chunk işlendi, "
        f"{len(merged_detail.get('keywords', []))} keyword, "
        f"{len(merged_detail.get('details', []))} detail toplandı."
    )

    # Katman 2+3: Reuse shared chunks for analyzer (avoid duplicate chunking)
    analyzer = ChunkAnalyzer(llm_client=llm)
    analyses = analyzer.analyze_all(detail_chunk_texts)
    print(f"[Chat | Katman 2+3] ✓ {len(analyses)} chunk analiz edildi.")

    # Katman 4: Merge
    merger = MergeLayer(llm_client=llm)
    merged = merger.merge(analyses)
    print("[Chat | Katman 4] ✓ Chunk analizleri birleştirildi.")

    # Katman 5: Final Memory
    final_memory_layer = FinalMemoryLayer(llm_client=llm)
    memory = final_memory_layer.generate(merged)
    print(f"[DEBUG] chat memory uzunluğu: {len(memory)}")
    print("[Chat | Katman 5] ✓ Chat hafızası oluşturuldu.")

    return memory, detail_memory


def run_code_pipeline(project_path: str, llm: LLMClient) -> tuple[str, str]:
    print("\n" + "=" * 55)
    print("  [KOD PİPELİNE] Başlıyor...")
    print("=" * 55)

    if not os.path.isdir(project_path):
        print(f"[HATA] '{project_path}' bir klasör değil, kod pipeline atlanıyor.")
        return "", ""

    # Katman 1: Code Input
    scanner = CodeInputLayer()
    try:
        files = scanner.scan(project_path)
    except ValueError as e:
        print(f"[Kod | Katman 1] HATA: {e}")
        return "", ""

    if not files:
        print("[Kod | Katman 1] HATA: Hiç kod dosyası bulunamadı.")
        return "", ""
    print(f"[Kod | Katman 1] ✓ {len(files)} dosya bulundu.")

    # Parallel code detail path (file-level chunks -> details/keywords)
    code_detail_extractor = CodeDetailExtractorLayer(llm_client=llm, debug=True)
    code_detail_merge = CodeDetailMergeLayer()
    code_extracted_details = code_detail_extractor.extract_all(files)
    code_merged_detail = code_detail_merge.merge(code_extracted_details, debug=True)
    code_detail_memory = code_merged_detail.get("memory_text", "")
    print(
        f"[Kod | Detail Path] ✓ {len(code_extracted_details)} dosya işlendi, "
        f"{len(code_merged_detail.get('variables', []))} variable/parameter toplandı."
    )

    # Katman 2: Code Analyzer
    analyzer = CodeAnalyzerLayer(llm_client=llm)
    analyses = analyzer.analyze_all(files)
    print(f"[Kod | Katman 2] ✓ {len(analyses)} dosya analiz edildi.")

    # Katman 3: Relation Layer
    relation_layer = CodeRelationLayer(llm_client=llm)
    relation_map   = relation_layer.map(analyses)
    print("[Kod | Katman 3] ✓ İlişki haritası oluşturuldu.")

    # Katman 4: Code Merge
    merge_layer = CodeMergeLayer(llm_client=llm)
    merged      = merge_layer.merge(analyses, relation_map)
    print("[Kod | Katman 4] ✓ Kod analizleri birleştirildi.")

    # Katman 5: Code Final Memory
    final_memory_layer = CodeFinalMemoryLayer(llm_client=llm)
    memory = final_memory_layer.generate(merged)
    print("[Kod | Katman 5] ✓ Kod hafızası oluşturuldu.")

    return memory, code_detail_memory


def main():
    if len(sys.argv) < 3:
        print("\nKullanım: python main_combined.py <chat.txt> <proje_klasörü>")
        print("Örnek   : python main_combined.py chat.txt /home/burak/Masaüstü/AgentSummarize")
        sys.exit(1)

    chat_file_path = sys.argv[1]
    project_path   = sys.argv[2]

    print("=" * 55)
    print("     COMBINED MEMORY SYSTEM — MVP")
    print("=" * 55)
    print(f"\n[Girdi] Chat dosyası : {chat_file_path}")
    print(f"[Girdi] Proje klasörü: {os.path.abspath(project_path)}")

    llm = LLMClient()

    chat_memory, detail_memory = run_chat_pipeline(chat_file_path, llm)
    code_memory, code_detail_memory = run_code_pipeline(project_path, llm)

    if not chat_memory and not code_memory and not detail_memory and not code_detail_memory:
        print("\n[HATA] Her iki pipeline da başarısız oldu.")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  [BİRLEŞTİRİCİ] Hafızalar birleştiriliyor...")
    print("=" * 55)

    combiner     = CombinedMemoryLayer(llm_client=llm)
    final_prompt = combiner.generate(chat_memory, code_memory, detail_memory, code_detail_memory)

    print("\n" + "=" * 55)
    print("         CHAT HAFIZASI (Ham)")
    print("=" * 55)
    print(f"\n{chat_memory}")

    print("\n" + "=" * 55)
    print("         KOD HAFIZASI (Ham)")
    print("=" * 55)
    print(f"\n{code_memory}")

    print("\n" + "=" * 55)
    print("         DETAY + KEYWORD HAFIZASI (Ham)")
    print("=" * 55)
    print(f"\n{detail_memory}")

    print("\n" + "=" * 55)
    print("         CODE VARIABLE/PARAMETER HAFIZASI (Ham)")
    print("=" * 55)
    print(f"\n{code_detail_memory}")

    print("\n" + "=" * 55)
    print("         FINAL BİRLEŞİK PROMPT")
    print("=" * 55)
    print(f"\n{final_prompt}")
    print("\n" + "=" * 55)
    print("Pipeline tamamlandı — tüm katmanlar aktif")
    print("=" * 55)


if __name__ == "__main__":
    main()