# Household Registration Document Processor

[简体中文](README.md)

An offline Windows OCR tool for scanned household registration PDFs. It treats one PDF as one household, aligns pages, extracts fields from household covers and member cards, and writes Excel sheets for member records, manual review, and processing logs. It creates separate PDF copies and preserves originals.

![Actual application window before documents are loaded](docs/screenshots/app.png)

## Run on Windows

Install Python 3.12 and run the following in PowerShell from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\fetch_models.py
.\.venv\Scripts\python.exe main.py
```

The setup step downloads public OCR weights and verifies them using SHA-256. Once configured, recognition and export run locally. Select a directory of documents you are authorized to process and review the flagged fields before using the export. The desktop UI and Excel headers are in Simplified Chinese.

## Implementation

- **Document processing:** PyMuPDF rasterizes pages, and OpenCV and NumPy handle orientation and layout.
- **Recognition:** RapidOCR with local PP-OCR weights and ONNX Runtime CPU inference. Uncertain fields go to a review sheet.
- **Batch work:** The scheduler supports configurable parallelism, pause and stop controls, and checkpoint-based resume.
- **Delivery:** PySide6 desktop UI, `parse` and `batch` CLI modes, openpyxl workbook exports, and a PyInstaller Windows build.

Run `.\.venv\Scripts\python.exe -m pytest -q` for the checks or `.\build.ps1` to build the Windows application. See the [Chinese README](README.md) for CLI examples and [third-party notices](docs/third-party.md) for model attribution. Recognition depends on document layout and scan quality and requires human review.
