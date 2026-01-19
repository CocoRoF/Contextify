# test.py
"""
DocumentProcessor Test Script - extract_chunks 테스트
"""
import logging
import sys
sys.path.insert(0, r"C:\Users\USER\Desktop\xgen\Contextify")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    from libs.core.document_processor import DocumentProcessor
    
    file_path = r"test\플래티어_전문연구요원_관리_규정.pdf"
    processor = DocumentProcessor()

    print("=" * 80)
    print("extract_chunks 테스트")
    print("=" * 80)

    result = processor.extract_chunks(
        file_path,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Total chunks: {len(result.chunks)}")

    for i, chunk in enumerate(result.chunks):
        print(f"\n--- Chunk {i + 1}/{len(result.chunks)} ---")
        print(f"Length: {len(chunk)} characters")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        print("-" * 40)

    # Save to markdown file
    saved_path = result.save_to_md("temp/", filename="chunks_output.md")
    print(f"\n✅ Saved chunks to: {saved_path}")


if __name__ == "__main__":
    main()
