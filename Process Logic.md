# Contextify Processing Flow

---

## Main Flow

```
User calls: processor.extract_chunks(file_path)
                    │
                    ▼
         DocumentProcessor.extract_chunks()
                    │
                    ├─► extract_text()
                    │       │
                    │       ├─► _create_current_file(file_path)
                    │       ├─► _get_handler(extension)
                    │       ├─► handler.extract_text(current_file)
                    │       └─► OCR processing (optional)
                    │
                    └─► chunk_text()
                            │
                            └─► create_chunks()
```

---

## PDF Handler Flow

```
PDFHandler.extract_text(current_file)
    │
    ├─► file_converter.convert(file_data)               [INTERFACE: PDFFileConverter]
    │       └─► Binary → fitz.Document
    │
    ├─► preprocessor.preprocess(doc)                    [INTERFACE: PDFPreprocessor]
    │       └─► Pass-through (returns PreprocessedData with doc unchanged)
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: PDFMetadataExtractor]
    │
    ├─► _extract_all_tables(doc, file_path)             [INTERNAL]
    │
    └─► For each page:
            │
            ├─► ComplexityAnalyzer.analyze()            [CLASS: pdf_complexity_analyzer]
            │       └─► Returns PageComplexity with recommended_strategy
            │
            ├─► Branch by strategy:
            │       │
            │       ├─► FULL_PAGE_OCR:
            │       │       └─► _process_page_full_ocr()
            │       │
            │       ├─► BLOCK_IMAGE_OCR:
            │       │       └─► _process_page_block_ocr()
            │       │
            │       ├─► HYBRID:
            │       │       └─► _process_page_hybrid()
            │       │
            │       └─► TEXT_EXTRACTION (default):
            │               └─► _process_page_text_extraction()
            │                       │
            │                       ├─► VectorTextOCREngine.detect_and_extract()
            │                       ├─► extract_text_blocks()           [FUNCTION]
            │                       ├─► format_image_processor methods  [INTERFACE: PDFImageProcessor]
            │                       └─► merge_page_elements()           [FUNCTION]
            │
            └─► page_tag_processor.create_page_tag()    [INTERFACE: PageTagProcessor]
```

---

## DOCX Handler Flow

```
DOCXHandler.extract_text(current_file)
    │
    ├─► file_converter.validate(file_data)              [INTERFACE: DOCXFileConverter]
    │       └─► Check if valid ZIP with [Content_Types].xml
    │
    ├─► If not valid DOCX:
    │       └─► _extract_with_doc_handler_fallback()    [INTERNAL]
    │               └─► DOCHandler.extract_text()       [DELEGATION]
    │
    ├─► file_converter.convert(file_data)               [INTERFACE: DOCXFileConverter]
    │       └─► Binary → docx.Document
    │
    ├─► preprocessor.preprocess(doc)                    [INTERFACE: DOCXPreprocessor]
    │       └─► Returns PreprocessedData (doc in extracted_resources)
    │
    ├─► chart_extractor.extract_all_from_file()         [INTERFACE: DOCXChartExtractor]
    │       └─► Pre-extract all charts (callback pattern)
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: DOCXMetadataExtractor]
    │
    └─► For each element in doc.element.body:
            │
            ├─► If paragraph ('p'):
            │       └─► process_paragraph_element()     [FUNCTION: docx_helper]
            │               ├─► format_image_processor.process_drawing_element()
            │               ├─► format_image_processor.extract_from_pict()
            │               └─► get_next_chart() callback for charts
            │
            └─► If table ('tbl'):
                    └─► process_table_element()         [FUNCTION: docx_helper]
```

---

## DOC Handler Flow

```
DOCHandler.extract_text(current_file)
    │
    ├─► file_converter.convert()                        [INTERFACE: DOCFileConverter]
    │       │
    │       ├─► _detect_format() → DocFormat (RTF/OLE/HTML/DOCX)
    │       │
    │       ├─► RTF: file_data (bytes) 반환             [Pass-through]
    │       ├─► OLE: _convert_ole() → olefile.OleFileIO
    │       ├─► HTML: _convert_html() → BeautifulSoup
    │       └─► DOCX: _convert_docx() → docx.Document
    │
    ├─► preprocessor.preprocess(converted_obj)          [INTERFACE: DOCPreprocessor]
    │       └─► Returns PreprocessedData (converted_obj in extracted_resources)
    │
    ├─► RTF format detected:
    │       └─► _delegate_to_rtf_handler()              [DELEGATION]
    │               └─► RTFHandler.extract_text(current_file)
    │
    ├─► OLE format detected:
    │       └─► _extract_from_ole_obj()                 [INTERNAL]
    │               ├─► _extract_ole_metadata()
    │               ├─► _extract_ole_text()
    │               └─► _extract_ole_images()
    │
    ├─► HTML format detected:
    │       └─► _extract_from_html_obj()                [INTERNAL]
    │               ├─► _extract_html_metadata()
    │               └─► BeautifulSoup parsing
    │
    └─► DOCX format detected:
            └─► _extract_from_docx_obj()                [INTERNAL]
                    └─► docx.Document paragraph/table extraction
```

---

## RTF Handler Flow

**구조**: Converter는 pass-through, Preprocessor에서 binary 처리, Handler에서 순차적 처리.

```
RTFHandler.extract_text(current_file)
    │
    ├─► file_converter.convert()                        [INTERFACE: RTFFileConverter]
    │       └─► Pass-through (returns raw bytes)
    │
    ├─► preprocessor.preprocess()                       [INTERFACE: RTFPreprocessor]
    │       │
    │       ├─► \binN tag processing (skip binary data)
    │       ├─► \pict group image extraction
    │       └─► Returns PreprocessedData (clean_content, image_tags, encoding)
    │
    ├─► decode_content()                                [FUNCTION: rtf_decoder]
    │       └─► bytes → string with detected encoding
    │
    ├─► Build RTFConvertedData                          [DATACLASS]
    │
    └─► _extract_from_converted()                       [INTERNAL]
            │
            ├─► metadata_extractor.extract()            [INTERFACE: RTFMetadataExtractor]
            ├─► metadata_extractor.format()
            │
            ├─► extract_tables_with_positions()         [FUNCTION: rtf_table_extractor]
            │
            ├─► extract_inline_content()                [FUNCTION: rtf_content_extractor]
            │
            └─► Build result string
```

---

## Excel Handler Flow (XLSX)

```
ExcelHandler.extract_text(current_file) [XLSX]
    │
    ├─► file_converter.convert(file_data, extension='xlsx')  [INTERFACE: ExcelFileConverter]
    │       └─► Binary → openpyxl.Workbook
    │
    ├─► preprocessor.preprocess(wb)                     [INTERFACE: ExcelPreprocessor]
    │       └─► Returns PreprocessedData (wb in extracted_resources)
    │
    ├─► _preload_xlsx_data()                            [INTERNAL]
    │       ├─► metadata_extractor.extract()            [INTERFACE: XLSXMetadataExtractor]
    │       ├─► chart_extractor.extract_all_from_file() [INTERFACE: ExcelChartExtractor]
    │       └─► format_image_processor.extract_images() [INTERFACE: ExcelImageProcessor]
    │
    └─► For each sheet:
            │
            ├─► _process_xlsx_sheet()                   [INTERNAL]
            │       ├─► page_tag_processor.create_sheet_tag()  [INTERFACE: PageTagProcessor]
            │       ├─► extract_textboxes_from_xlsx()   [FUNCTION]
            │       ├─► convert_xlsx_sheet_to_table()   [FUNCTION]
            │       └─► convert_xlsx_objects_to_tables()[FUNCTION]
            │
            └─► format_image_processor.get_sheet_images()  [INTERFACE: ExcelImageProcessor]
```

---

## Excel Handler Flow (XLS)

```
ExcelHandler.extract_text(current_file) [XLS]
    │
    ├─► file_converter.convert(file_data, extension='xls')   [INTERFACE: ExcelFileConverter]
    │       └─► Binary → xlrd.Book
    │
    ├─► preprocessor.preprocess(wb)                     [INTERFACE: ExcelPreprocessor]
    │       └─► Returns PreprocessedData (wb in extracted_resources)
    │
    ├─► _get_xls_metadata_extractor().extract_and_format()   [INTERFACE: XLSMetadataExtractor]
    │
    └─► For each sheet:
            │
            ├─► page_tag_processor.create_sheet_tag()   [INTERFACE: PageTagProcessor]
            │
            ├─► convert_xls_sheet_to_table()            [FUNCTION]
            │
            └─► convert_xls_objects_to_tables()         [FUNCTION]
```

---

## PPT Handler Flow

```
PPTHandler.extract_text(current_file)
    │
    ├─► file_converter.convert(file_data, file_stream)  [INTERFACE: PPTFileConverter]
    │       └─► Binary → pptx.Presentation
    │
    ├─► preprocessor.preprocess(prs)                    [INTERFACE: PPTPreprocessor]
    │       └─► Returns PreprocessedData (prs in extracted_resources)
    │
    ├─► chart_extractor.extract_all_from_file()         [INTERFACE: PPTChartExtractor]
    │       └─► Pre-extract all charts (callback pattern)
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: PPTMetadataExtractor]
    ├─► metadata_extractor.format()                     [INTERFACE: PPTMetadataExtractor]
    │
    └─► For each slide:
            │
            ├─► page_tag_processor.create_slide_tag()   [INTERFACE: PageTagProcessor]
            │
            └─► For each shape:
                    │
                    ├─► If table: convert_table_to_html()       [FUNCTION]
                    ├─► If chart: get_next_chart() callback     [Pre-extracted]
                    ├─► If picture: process_image_shape()       [FUNCTION]
                    ├─► If group: process_group_shape()         [FUNCTION]
                    └─► If text: extract_text_with_bullets()    [FUNCTION]
```

---

## HWP Handler Flow

```
HWPHandler.extract_text(current_file)
    │
    ├─► file_converter.validate(file_data)              [INTERFACE: HWPFileConverter]
    │       └─► Check if OLE file (magic number check)
    │
    ├─► If not OLE file:
    │       └─► _handle_non_ole_file()                  [INTERNAL]
    │               ├─► ZIP detected → HWPXHandler delegation
    │               └─► HWP 3.0 → Not supported
    │
    ├─► chart_extractor.extract_all_from_file()         [INTERFACE: HWPChartExtractor]
    │
    ├─► file_converter.convert()                        [INTERFACE: HWPFileConverter]
    │       └─► Binary → olefile.OleFileIO
    │
    ├─► preprocessor.preprocess(ole)                    [INTERFACE: HWPPreprocessor]
    │       └─► Returns PreprocessedData (ole in extracted_resources)
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: HWPMetadataExtractor]
    ├─► metadata_extractor.format()                     [INTERFACE: HWPMetadataExtractor]
    │
    ├─► _parse_docinfo(ole)                             [INTERNAL]
    │       └─► parse_doc_info()                        [FUNCTION]
    │
    ├─► _extract_body_text(ole)                         [INTERNAL]
    │       │
    │       └─► For each section:
    │               ├─► decompress_section()            [FUNCTION]
    │               └─► _parse_section()                [INTERNAL]
    │                       └─► _process_picture()      [INTERNAL - format_image_processor 사용]
    │
    ├─► format_image_processor.process_images_from_bindata()  [INTERFACE: HWPImageProcessor]
    │
    └─► file_converter.close(ole)                       [INTERFACE: HWPFileConverter]
```

---

## HWPX Handler Flow

```
HWPXHandler.extract_text(current_file)
    │
    ├─► get_file_stream(current_file)                   [INHERITED: BaseHandler]
    │       └─► BytesIO(file_data)
    │
    ├─► _is_valid_zip(file_stream)                      [INTERNAL]
    │
    ├─► chart_extractor.extract_all_from_file()         [INTERFACE: HWPXChartExtractor]
    │
    ├─► zipfile.ZipFile(file_stream)                    [EXTERNAL LIBRARY]
    │
    ├─► preprocessor.preprocess(zf)                     [INTERFACE: HWPXPreprocessor]
    │       └─► Returns PreprocessedData (extracted_resources available)
    │
    ├─► metadata_extractor.extract()                    [INTERFACE: HWPXMetadataExtractor]
    ├─► metadata_extractor.format()                     [INTERFACE: HWPXMetadataExtractor]
    │
    ├─► parse_bin_item_map(zf)                          [FUNCTION]
    │
    ├─► For each section:
    │       │
    │       └─► parse_hwpx_section()                    [FUNCTION]
    │               │
    │               ├─► format_image_processor.process_images()  [INTERFACE: HWPXImageProcessor]
    │               │
    │               └─► parse_hwpx_table()              [FUNCTION]
    │
    └─► format_image_processor.get_remaining_images()   [INTERFACE: HWPXImageProcessor]
        format_image_processor.process_images()         [INTERFACE: HWPXImageProcessor]
```

---

## CSV Handler Flow

```
CSVHandler.extract_text(current_file)
    │
    ├─► file_converter.convert(file_data, encoding)     [INTERFACE: CSVFileConverter]
    │       └─► Binary → Text (with encoding detection)
    │
    ├─► preprocessor.preprocess(content)                [INTERFACE: CSVPreprocessor]
    │       └─► Returns PreprocessedData (content in clean_content)
    │
    ├─► detect_delimiter(content)                       [FUNCTION]
    │
    ├─► parse_csv_content(content, delimiter)           [FUNCTION]
    │
    ├─► detect_header(rows)                             [FUNCTION]
    │
    ├─► metadata_extractor.extract(source_info)         [INTERFACE: CSVMetadataExtractor]
    │       └─► CSVSourceInfo contains: file_path, encoding, delimiter, rows, has_header
    │
    └─► convert_rows_to_table(rows, has_header)         [FUNCTION]
            └─► Returns HTML table
```

---

## Text Handler Flow

```
TextHandler.extract_text(current_file)
    │
    ├─► preprocessor.preprocess(file_data)              [INTERFACE: TextPreprocessor]
    │       └─► Returns PreprocessedData (file_data in clean_content)
    │
    ├─► file_data.decode(encoding)                      [DIRECT: No FileConverter used]
    │       └─► Try encodings: utf-8, utf-8-sig, cp949, euc-kr, latin-1, ascii
    │
    └─► clean_text() / clean_code_text()                [FUNCTION: utils.py]
```

Note: TextHandler는 file_converter를 사용하지 않고 직접 decode합니다.

---

## HTML Handler Flow

```
HTMLReprocessor (Utility - NOT a BaseHandler subclass)
    │
    ├─► clean_html_file(html_content)                   [FUNCTION]
    │       │
    │       ├─► BeautifulSoup parsing
    │       ├─► Remove unwanted tags (script, style, etc.)
    │       ├─► Remove style attributes
    │       ├─► _process_table_merged_cells()
    │       └─► Return cleaned HTML string
    │
    └─► Used by DOCHandler when HTML format detected
```

Note: HTML은 별도의 BaseHandler 서브클래스가 없습니다.
      DOCHandler가 HTML 형식을 감지하면 내부적으로 BeautifulSoup으로 처리합니다.

---

## Image File Handler Flow

```
ImageFileHandler.extract_text(current_file)
    │
    ├─► preprocessor.preprocess(file_data)              [INTERFACE: ImageFilePreprocessor]
    │       └─► Returns PreprocessedData (file_data in clean_content)
    │
    ├─► Validate file extension                         [INTERNAL]
    │       └─► SUPPORTED_IMAGE_EXTENSIONS: jpg, jpeg, png, gif, bmp, webp
    │
    ├─► If OCR engine is None:
    │       └─► _build_image_tag(file_path)             [INTERNAL]
    │               └─► Return [image:path] tag
    │
    └─► If OCR engine available:
            └─► _ocr_engine.extract_text()              [INTERFACE: BaseOCR]
                    └─► Image → Text via OCR
```

Note: ImageFileHandler는 OCR 엔진이 설정된 경우에만 실제 텍스트 추출이 가능합니다.

---

## Chunking Flow

```
chunk_text(text, chunk_size, chunk_overlap)
    │
    └─► create_chunks()                                 [FUNCTION]
            │
            ├─► _extract_document_metadata()            [FUNCTION]
            │
            ├─► Detect file type:
            │       │
            │       ├─► Table-based (xlsx, xls, csv):
            │       │       └─► chunk_multi_sheet_content()  [FUNCTION]
            │       │
            │       ├─► Text with page markers:
            │       │       └─► chunk_by_pages()        [FUNCTION]
            │       │
            │       └─► Plain text:
            │               └─► chunk_plain_text()      [FUNCTION]
            │
            └─► _prepend_metadata_to_chunks()           [FUNCTION]
```

---

## Interface Integration Summary

```
┌─────────────┬─────────────────────┬─────────────────────┬─────────────────────┬─────────────────────┬─────────────────────┐
│ Handler     │ FileConverter       │ Preprocessor        │ MetadataExtractor   │ ChartExtractor      │ FormatImageProcessor│
├─────────────┼─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ PDF         │ ✅ PDFFileConverter  │ ✅ PDFPreprocessor   │ ✅ PDFMetadata       │ ❌ NullChart         │ ✅ PDFImage          │
│ DOCX        │ ✅ DOCXFileConverter │ ✅ DOCXPreprocessor  │ ✅ DOCXMetadata      │ ✅ DOCXChart         │ ✅ DOCXImage         │
│ DOC         │ ✅ DOCFileConverter  │ ✅ DOCPreprocessor   │ ❌ NullMetadata      │ ❌ NullChart         │ ✅ DOCImage          │
│ RTF         │ ✅ RTFFileConverter  │ ✅ RTFPreprocessor*  │ ✅ RTFMetadata       │ ❌ NullChart         │ ❌ Uses base         │
│ XLSX        │ ✅ ExcelFileConverter│ ✅ ExcelPreprocessor │ ✅ XLSXMetadata      │ ✅ ExcelChart        │ ✅ ExcelImage        │
│ XLS         │ ✅ ExcelFileConverter│ ✅ ExcelPreprocessor │ ✅ XLSMetadata       │ ✅ ExcelChart        │ ✅ ExcelImage        │
│ PPT/PPTX    │ ✅ PPTFileConverter  │ ✅ PPTPreprocessor   │ ✅ PPTMetadata       │ ✅ PPTChart          │ ✅ PPTImage          │
│ HWP         │ ✅ HWPFileConverter  │ ✅ HWPPreprocessor   │ ✅ HWPMetadata       │ ✅ HWPChart          │ ✅ HWPImage          │
│ HWPX        │ ❌ None (직접 ZIP)   │ ✅ HWPXPreprocessor  │ ✅ HWPXMetadata      │ ✅ HWPXChart         │ ✅ HWPXImage         │
│ CSV         │ ✅ CSVFileConverter  │ ✅ CSVPreprocessor   │ ✅ CSVMetadata       │ ❌ NullChart         │ ✅ CSVImage          │
│ TXT/MD/JSON │ ❌ None (직접 decode)│ ✅ TextPreprocessor  │ ❌ NullMetadata      │ ❌ NullChart         │ ✅ TextImage         │
│ HTML        │ ❌ N/A (유틸리티)    │ ❌ N/A               │ ❌ N/A               │ ❌ N/A               │ ❌ N/A               │
│ Image Files │ ✅ ImageFileConverter│ ✅ ImagePreprocessor │ ❌ NullMetadata      │ ❌ NullChart         │ ✅ ImageFileImage    │
└─────────────┴─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘

✅ = Interface implemented
❌ = Not applicable / NullExtractor / Not used
* = RTFPreprocessor has actual processing logic (image extraction, binary cleanup)
```

---

## Handler Processing Pipeline

모든 핸들러는 동일한 처리 파이프라인을 따릅니다:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           Handler Processing Pipeline                             │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  1. FileConverter.convert()     Binary → Format-specific object                  │
│         │                       (fitz.Document, docx.Document, olefile, etc.)    │
│         ▼                                                                         │
│  2. Preprocessor.preprocess()   Process/clean the converted data                 │
│         │                       (image extraction, binary cleanup, encoding)     │
│         ▼                                                                         │
│  3. MetadataExtractor.extract() Extract document metadata                        │
│         │                       (title, author, created date, etc.)              │
│         ▼                                                                         │
│  4. Content Extraction          Format-specific content extraction               │
│         │                       (text, tables, images, charts)                   │
│         ▼                                                                         │
│  5. Result Assembly             Build final result string                        │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘

Note: 대부분의 핸들러에서 Preprocessor는 pass-through (NullPreprocessor).
      RTF는 예외로, RTFPreprocessor에서 실제 바이너리 처리가 이루어짐.
```

---

## Remaining Function-Based Components

```
┌─────────────┬────────────────────────────────────────────────────────────┐
│ Handler     │ Function-Based Components                                  │
├─────────────┼────────────────────────────────────────────────────────────┤
│ PDF         │ extract_text_blocks(), merge_page_elements(),             │
│             │ ComplexityAnalyzer, VectorTextOCREngine,                  │
│             │ BlockImageEngine                                          │
├─────────────┼────────────────────────────────────────────────────────────┤
│ DOCX        │ process_paragraph_element(), process_table_element()      │
├─────────────┼────────────────────────────────────────────────────────────┤
│ DOC         │ Format detection, OLE/HTML/DOCX internal processing       │
├─────────────┼────────────────────────────────────────────────────────────┤
│ RTF         │ decode_content() (rtf_decoder.py)                         │
│             │ extract_tables_with_positions() (rtf_table_extractor.py)  │
│             │ extract_inline_content() (rtf_content_extractor.py)       │
├─────────────┼────────────────────────────────────────────────────────────┤
│ Excel       │ extract_textboxes_from_xlsx(), convert_xlsx_sheet_to_table│
│             │ convert_xls_sheet_to_table(), convert_*_objects_to_tables │
├─────────────┼────────────────────────────────────────────────────────────┤
│ PPT         │ extract_text_with_bullets(), convert_table_to_html(),     │
│             │ process_image_shape(), process_group_shape()              │
├─────────────┼────────────────────────────────────────────────────────────┤
│ HWP         │ parse_doc_info(), decompress_section()                    │
├─────────────┼────────────────────────────────────────────────────────────┤
│ HWPX        │ parse_bin_item_map(), parse_hwpx_section()                │
├─────────────┼────────────────────────────────────────────────────────────┤
│ CSV         │ detect_delimiter(), parse_csv_content(), detect_header(), │
│             │ convert_rows_to_table()                                   │
├─────────────┼────────────────────────────────────────────────────────────┤
│ Text        │ clean_text(), clean_code_text() (utils.py)                │
├─────────────┼────────────────────────────────────────────────────────────┤
│ HTML        │ clean_html_file(), _process_table_merged_cells()          │
│             │ (html_reprocessor.py - utility, not handler)              │
├─────────────┼────────────────────────────────────────────────────────────┤
│ Image       │ OCR engine integration (BaseOCR subclass)                 │
├─────────────┼────────────────────────────────────────────────────────────┤
│ Chunking    │ create_chunks(), chunk_by_pages(), chunk_plain_text(),    │
│             │ chunk_multi_sheet_content(), chunk_large_table()          │
└─────────────┴────────────────────────────────────────────────────────────┘
```
