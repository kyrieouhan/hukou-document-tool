"""OCR interface with an explicitly local model implementation."""
from dataclasses import dataclass
from typing import Protocol
from pathlib import Path
import numpy as np
from .runtime import thread_settings

@dataclass
class Token:
    box: list
    text: str
    confidence: float

    @property
    def x(self) -> float:
        return sum(p[0] for p in self.box)/4

    @property
    def y(self) -> float:
        return sum(p[1] for p in self.box)/4

class OcrEngine(Protocol):
    def read(self, image: np.ndarray) -> list[Token]: ...
    def read_line(self, image: np.ndarray) -> tuple[str,float]: ...

class RapidEngine:
    def __init__(self) -> None:
        import onnxruntime
        onnxruntime.disable_telemetry_events()
        import rapidocr_onnxruntime
        from rapidocr_onnxruntime import RapidOCR
        folder = Path(rapidocr_onnxruntime.__file__).parent/'models'
        names = ['ch_PP-OCRv4_det_infer.onnx','ch_ppocr_mobile_v2.0_cls_infer.onnx',
                 'ch_PP-OCRv4_rec_infer.onnx']
        paths = [folder/n for n in names]
        if not all(p.is_file() for p in paths):
            raise RuntimeError('本地 OCR 模型不完整，请重新解压完整程序文件夹。')
        threads=thread_settings()
        self.impl = RapidOCR(det_model_path=str(paths[0]),cls_model_path=str(paths[1]),
                             rec_model_path=str(paths[2]), intra_op_num_threads=threads.onnx_intra,
                             inter_op_num_threads=threads.onnx_inter, det_limit_side_len=1600,det_limit_type='max',
                             det_use_cuda=False,cls_use_cuda=False,rec_use_cuda=False,
                             det_use_dml=False,cls_use_dml=False,rec_use_dml=False)

    def read(self, image: np.ndarray) -> list[Token]:
        if image.size == 0:
            return []
        result,_ = self.impl(image, use_cls=False)
        return [Token(b,t,float(c)) for b,t,c in (result or [])]

    def read_line(self,image: np.ndarray) -> tuple[str,float]:
        result,_=self.impl(image,use_det=False,use_cls=False)
        return (result[0][0],float(result[0][1])) if result else ('',0.)

class PrecisionRecognizer:
    """Local PP-OCRv5 recognition for small fields; orientation retains V1 OCR."""
    def __init__(self,model_name: str='ch_PP-OCRv5_rec_server.onnx') -> None:
        import onnxruntime
        from rapidocr_onnxruntime.ch_ppocr_rec.text_recognize import TextRecognizer
        onnxruntime.disable_telemetry_events()
        folder=Path(__file__).parent/'ocr_models'
        model=folder/model_name
        if not model.is_file():raise RuntimeError('本地字段识别模型缺失，请重新解压完整程序。')
        threads=thread_settings()
        self.impl=TextRecognizer(dict(model_path=str(model),
              rec_img_shape=[3,48,320],rec_batch_num=6,intra_op_num_threads=threads.onnx_intra,
              inter_op_num_threads=threads.onnx_inter,use_cuda=False,use_dml=False))
        classes=self.impl.session.session.get_outputs()[0].shape[-1]
        if isinstance(classes,int) and classes!=len(self.impl.postprocess_op.character):
            raise RuntimeError('识别模型字符表不匹配，请重新解压程序。')

    def read_lines(self,images: list[np.ndarray]) -> list[tuple[str,float]]:
        if not images:return []
        result,_=self.impl(images)
        return [(str(v),float(c)) for v,c in result]

    def read_line(self,image: np.ndarray) -> tuple[str,float]:
        return self.read_lines([image])[0] if image.size else ('',0.)

    def read_aligned(self,image: np.ndarray) -> list[dict]:
        """CTC character positions locate printed groups without correcting text."""
        result,_=self.impl(image,return_word_box=True)
        if not result:return []
        text,confidence,info=result[0]
        steps,words,columns,states,scores=info
        groups=[];offset=0
        for word,positions in zip(words,columns):
            count=len(word)
            groups.append(dict(text=''.join(word),positions=[float(p)/steps*image.shape[1] for p in positions],
                               confidence=float(np.mean(scores[offset:offset+count]))))
            offset+=count
        return groups
