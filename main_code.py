# main_code.py

import sys
import os
from code_layers.code_input_layer import CodeInputLayer
from code_layers.code_analyzer import CodeAnalyzerLayer
from code_layers.code_relation_layer import CodeRelationLayer
from code_layers.code_merge_layer import CodeMergeLayer
from code_layers.code_final_memory import CodeFinalMemoryLayer
from layers.llm_client import LLMClient


def main():
    project_path = sys.argv[1] if len(sys.argv) > 1 else "."

    print("=" * 55)
    print("        CODE MEMORY SYSTEM — MVP")
    print("=" * 55)
    print(f"\n[Girdi] Proje klasörü: {os.path.abspath(project_path)}")

    if not os.path.isdir(project_path):
        print(f"\n[HATA] '{project_path}' bir klasör değil!")
        sys.exit(1)

    # ----------------------------------------------------------------
    # KATMAN 1: Code Input Layer
    # ----------------------------------------------------------------
    print("\n[Katman 1] Proje taranıyor...")
    scanner = CodeInputLayer()

    try:
        files = scanner.scan(project_path)
    except ValueError as e:
        print(f"\n[HATA] {e}")
        sys.exit(1)

    if not files:
        print("\n[HATA] Hiç kod dosyası bulunamadı!")
        sys.exit(1)

    print(f"[Katman 1] ✓ {len(files)} dosya bulundu.")
    print("\n--- Bulunan Dosyalar ---")
    for f in files:
        truncated_flag = " [KISALTILDI]" if f["truncated"] else ""
        print(f"  [{f['index']:02d}] {f['path']:<45} {f['size_chars']:>6} karakter{truncated_flag}")

    # ----------------------------------------------------------------
    # KATMAN 2: Code Analyzer
    # ----------------------------------------------------------------
    print("\n[Katman 2] Dosyalar analiz ediliyor...")
    llm      = LLMClient()
    analyzer = CodeAnalyzerLayer(llm_client=llm)
    analyses = analyzer.analyze_all(files)

    print(f"\n[Katman 2] ✓ {len(analyses)} dosya analiz edildi.")
    print("\n--- Dosya Analizleri ---")
    for a in analyses:
        print(f"\n  {a['file']}")
        print(f"    Amaç         : {a['purpose']}")
        print(f"    Class'lar    : {a['classes']}")
        print(f"    Fonksiyonlar : {a['functions']}")
        print(f"    Bağımlılıklar: {a['dependencies']}")
        if a["notes"]:
            print(f"    Notlar       : {a['notes']}")

    # ----------------------------------------------------------------
    # KATMAN 3: Relation Layer
    # ----------------------------------------------------------------
    print("\n[Katman 3] Dosyalar arası ilişkiler analiz ediliyor...")
    relation_layer = CodeRelationLayer(llm_client=llm)
    relation_map   = relation_layer.map(analyses)

    print("\n[Katman 3] ✓ İlişki haritası oluşturuldu.")
    print("\n--- İlişki Haritası ---")
    print(f"\n  Mimari    : {relation_map.get('architecture', '')}")
    print(f"  Hub'lar   : {relation_map.get('hubs', [])}")
    print(f"  Giriş Nokt: {relation_map.get('entry_points', [])}")
    print(f"  Core Mod. : {relation_map.get('core_modules', [])}")
    print(f"\n  İlişkiler :")
    for r in relation_map.get("relations", []):
        print(f"    • {r}")

    # ----------------------------------------------------------------
    # KATMAN 4: Code Merge Layer
    # ----------------------------------------------------------------
    print("\n[Katman 4] Özetler birleştiriliyor...")
    merge_layer = CodeMergeLayer(llm_client=llm)
    merged      = merge_layer.merge(analyses, relation_map)

    print("\n[Katman 4] ✓ Birleştirildi.")
    print("\n--- Birleşik Proje Özeti ---")
    print(f"\n  Proje Adı  : {merged.get('project_name', '')}")
    print(f"  Amaç       : {merged.get('purpose', '')}")
    print(f"  Hub'lar    : {merged.get('hubs', [])}")
    print(f"  Core Mod.  : {merged.get('core_modules', [])}")
    print(f"  Giriş Nokt.: {merged.get('entry_points', [])}")

    # ----------------------------------------------------------------
    # KATMAN 5: Code Final Memory
    # ----------------------------------------------------------------
    print("\n[Katman 5] Final hafıza oluşturuluyor...")
    final_memory_layer = CodeFinalMemoryLayer(llm_client=llm)
    memory = final_memory_layer.generate(merged)

    print("\n[Katman 5] ✓ Hafıza oluşturuldu.")
    print("\n" + "=" * 55)
    print("           FINAL KOD HAFIZASI")
    print("=" * 55)
    print(f"\n{memory}")
    print("\n" + "=" * 55)
    print("Pipeline tamamlandı — tüm katmanlar aktif")
    print("=" * 55)


if __name__ == "__main__":
    main()