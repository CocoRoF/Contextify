# test_all_handlers.py
"""
DocumentProcessor - All Handlers Test Script

Tests all available handlers in the document processor.
"""
import logging
import sys
import os
import traceback
from pathlib import Path

sys.path.insert(0, r"C:\Users\USER\Desktop\xgen\Contextify")

logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test file configurations
TEST_FILES = {
    # PDF Handler
    'pdf': r'test\플래티어_전문연구요원_관리_규정.pdf',
    
    # Excel Handler
    'xlsx': r'test\sample.xlsx',
    
    # Text Handler (text, code, config, etc.)
    'txt': r'test\sample.txt',
    'csv': r'test\sample.csv',
    'html': r'test\sample.html',
    'json': r'test\sample.json',
    'md': r'test\sample.md',
}

# Handlers that need actual files (not available in test folder)
UNAVAILABLE_HANDLERS = {
    'docx': 'No .docx file in test folder',
    'doc': 'No .doc file in test folder', 
    'ppt': 'No .ppt file in test folder',
    'pptx': 'No .pptx file in test folder',
    'xls': 'No .xls file in test folder',
    'hwp': 'No .hwp file in test folder',
    'hwpx': 'No .hwpx file in test folder',
    'tsv': 'No .tsv file in test folder',
}


def test_handler(processor, ext: str, file_path: str) -> dict:
    """
    Test a single handler.
    
    Returns:
        dict with 'success', 'error', 'text_length', 'chunk_count'
    """
    result = {
        'extension': ext,
        'file': file_path,
        'success': False,
        'error': None,
        'text_length': 0,
        'chunk_count': 0,
        'sample_text': ''
    }
    
    try:
        if not os.path.exists(file_path):
            result['error'] = f"File not found: {file_path}"
            return result
        
        # Test extract_text
        text = processor.extract_text(file_path)
        result['text_length'] = len(text) if text else 0
        result['sample_text'] = (text[:150] + '...') if text and len(text) > 150 else (text or '')
        
        # Test extract_chunks
        chunk_result = processor.extract_chunks(file_path, chunk_size=500, chunk_overlap=50)
        result['chunk_count'] = len(chunk_result.chunks)
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
        result['traceback'] = traceback.format_exc()
    
    return result


def main():
    from libs.core.document_processor import DocumentProcessor
    
    processor = DocumentProcessor()
    
    print("=" * 80)
    print("DocumentProcessor - All Handlers Test")
    print("=" * 80)
    
    results = []
    
    # Test available handlers
    print("\n📁 Testing available handlers...\n")
    for ext, file_path in TEST_FILES.items():
        print(f"  Testing .{ext} handler... ", end='', flush=True)
        result = test_handler(processor, ext, file_path)
        results.append(result)
        
        if result['success']:
            print(f"✅ OK (text: {result['text_length']} chars, chunks: {result['chunk_count']})")
        else:
            print(f"❌ FAILED")
            print(f"      Error: {result['error']}")
    
    # Report unavailable handlers
    print("\n📋 Unavailable handlers (no test files):")
    for ext, reason in UNAVAILABLE_HANDLERS.items():
        print(f"  ⚠️  .{ext}: {reason}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    
    print(f"\n  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⚠️  Unavailable: {len(UNAVAILABLE_HANDLERS)}")
    
    # Show failed details
    if failed > 0:
        print("\n" + "-" * 80)
        print("FAILED HANDLERS DETAILS:")
        print("-" * 80)
        for r in results:
            if not r['success']:
                print(f"\n  [{r['extension']}] {r['file']}")
                print(f"  Error: {r['error']}")
                if 'traceback' in r:
                    print("  Traceback:")
                    for line in r['traceback'].split('\n')[-10:]:
                        if line.strip():
                            print(f"    {line}")
    
    # Show sample output
    print("\n" + "-" * 80)
    print("SAMPLE OUTPUT (first 150 chars of each):")
    print("-" * 80)
    for r in results:
        if r['success'] and r['sample_text']:
            print(f"\n  [{r['extension']}]:")
            sample = r['sample_text'].replace('\n', ' ')[:100]
            print(f"    {sample}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
