# Windows 便携版的第三方声明与对应源码

项目自有程序源码按 [AGPL-3.0](../LICENSE) 发布。随包的第三方组件保留各自的版权和许可。ZIP 内的 `licenses` 目录包含可随包分发的许可证和声明。

## 对应源码

- 本程序、本版依赖清单与构建步骤见 GitHub Release 的 `v1.2.0` 源码和 [README](../README.md)。
- PyMuPDF 1.28.2 与 MuPDF 1.28.2 按 AGPL-3.0 分发。对应源码和上游构建说明见 [PyMuPDF 1.28.2 源码包](https://pypi.org/project/PyMuPDF/1.28.2/#files) 和 [MuPDF 1.28.2 源码包](https://mupdf.com/downloads/archive/mupdf-1.28.2-source.tar.gz)。
- Qt 6.11.2、PySide6 和 Shiboken 6.11.2 使用上游的未修改动态库与插件，可通过 ZIP 中的 `_internal` 更换 ABI 兼容版本。源代码见 [Qt 6.11.2 模块源码](https://download.qt.io/archive/qt/6.11/6.11.2/submodules/) 和 [PySide6 / Shiboken 6.11.2 源码](https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.11.2-src/)。
- [CPython 3.12.5 对应源码](https://www.python.org/downloads/release/python-3125/) · [OpenCV 5.0.0 Python 构建源码](https://github.com/opencv/opencv-python/tree/93) · [PyInstaller 6.22.3 源码](https://github.com/pyinstaller/pyinstaller/tree/v6.22.3)。
- [ONNX Runtime 构建与源码说明](https://onnxruntime.ai/docs/build/) · [RapidOCR 源码](https://github.com/RapidAI/RapidOCR) · [模型许可与归属](https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/)。

## 模型

OCR 模型均为上游发布的通用预训练模型。三个字段识别模型使用 RapidOCR 的 v3.9.2 文件，准确文件名及 SHA-256 见 [模型下载脚本](../scripts/fetch_models.py)。用于检测和基础识别的 PP-OCRv4 模型由 `rapidocr-onnxruntime` 1.4.4 提供。上游模型由 PaddlePaddle/PaddleOCR 发布，许可为 [Apache-2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)。
