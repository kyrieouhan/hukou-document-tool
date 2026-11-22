# 第三方组件与 OCR 模型

程序依赖由各上游维护，采用各自的许可证。OCR 权重通过安装脚本从上游下载，软件处理 PDF 时使用本机文件。

## OCR 模型

`scripts/fetch_models.py` 固定使用 RapidOCR 模型仓库的 v3.9.2。模型来自 PaddlePaddle/PaddleOCR，ONNX 文件由 RapidOCR 提供，许可为 Apache-2.0。

| 文件 | 上游 |
| --- | --- |
| `ch_PP-OCRv5_rec_server.onnx` | [RapidOCR PP-OCRv5](https://www.modelscope.cn/models/RapidAI/RapidOCR/files) |
| `en_PP-OCRv5_rec_mobile.onnx` | [RapidOCR PP-OCRv5](https://www.modelscope.cn/models/RapidAI/RapidOCR/files) |
| `PP-OCRv6_rec_medium.onnx` | [RapidOCR PP-OCRv6](https://www.modelscope.cn/models/RapidAI/RapidOCR/files) |

用于页面文字定位的 PP-OCRv4 模型由 `rapidocr-onnxruntime` 依赖提供。

- [RapidOCR 模型许可与归属](https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/)
- [PaddleOCR 许可证](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)
- [PyMuPDF 授权说明](https://pymupdf.io/licensing)
- [Qt for Python 授权说明](https://doc.qt.io/qtforpython-6/licenses.html)
- [PyInstaller 授权说明](https://pyinstaller.org/en/stable/license.html)
