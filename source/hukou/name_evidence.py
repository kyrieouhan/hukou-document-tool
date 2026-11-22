"""Observed text-strip evidence for an interior character omitted by one model."""
from collections import Counter
import cv2
import numpy as np
from .imaging import flatten_background
from .ocr import OcrEngine,PrecisionRecognizer


def includes_interior(short: str,long: str) -> bool:
    """Do not confuse adjacent labels or prefixes with a missing interior glyph."""
    return len(long)==len(short)+1 and any(long[:i]+long[i+1:]==short for i in range(1,len(long)-1))


def verify_omission(image: np.ndarray,short: str,candidates: list[dict],base: OcrEngine,
                    recognizer: PrecisionRecognizer) -> tuple[str,list[dict]]:
    """Choose only an actual complete OCR string; caller must retain review."""
    from .candidates import normalize
    if not any(includes_interior(short,c['value']) and c['confidence']>=.70 for c in candidates):return short,[]
    impl=getattr(base,'impl',None)
    if impl is None:return short,[]
    boxes,_=impl.auto_text_det(image)
    if boxes is None or len(boxes)!=1:return short,[]
    strip=impl.get_crop_img_list(image,boxes)[0]
    if strip.shape[1]/strip.shape[0]<len(short)*.95:return short,[]
    views=[flatten_background(strip),cv2.GaussianBlur(strip,(0,0),1.),cv2.GaussianBlur(strip,(0,0),1.5)]
    observed=[dict(value=normalize('name',v),confidence=c,mode=mode,box=np.asarray(boxes[0]).tolist())
              for mode,(v,c) in zip(('flattened','light','descreen'),recognizer.read_lines(views))]
    votes=Counter(v['value'] for v in observed if v['confidence']>=.75 and includes_interior(short,v['value']))
    if votes and votes.most_common(1)[0][1]>=2:return votes.most_common(1)[0][0],observed
    return short,observed
