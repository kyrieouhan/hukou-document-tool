"""Second geometry view for weak, conflicting household-cover name readings."""
from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
import numpy as np
from .imaging import flatten_background,enhance
from .field_cells import remove_lines
from .ocr import OcrEngine
if TYPE_CHECKING:
    from .candidates import Resolution,FieldCandidate


def padded_name_candidates(image: np.ndarray,previous: Resolution,engines: dict[str,OcrEngine],page: int,index: int) -> list[FieldCandidate]:
    """Retain only evidence for an already observed name, with two-model support.

    A 30% text-height margin protects strokes lost at a tight detected boundary.
    It is within the already located value cell, never a PDF-page coordinate.
    Caller keeps LOW when this evidence changes the selected value.
    """
    from .candidates import FieldCandidate,resolve
    if previous.tier!='LOW':return []
    values={c.value for c in previous.candidates if c.confidence>=.60 and c.validity_score and 2<=len(c.value)<=24}
    if len(values)<2:return []
    impl=getattr(engines['v4'],'impl',None)
    if impl is None:return []
    boxes,_=impl.auto_text_det(image)
    if boxes is None or len(boxes)!=1:return []
    x1,y1=np.min(boxes[0],axis=0);x2,y2=np.max(boxes[0],axis=0);height=y2-y1
    crop=image[max(0,int(y1-height*.30)):min(image.shape[0],int(y2+height*.30)),max(0,int(x1)):min(image.shape[1],int(x2))]
    if not crop.size:return []
    variants={'original':crop,'light':cv2.GaussianBlur(crop,(0,0),1.),'flat':flatten_background(crop),'clahe':enhance(crop,'clahe'),'no_lines':remove_lines(crop)}
    candidates=[]
    for name,engine in engines.items():
        readings=engine.read_lines(list(variants.values())) if hasattr(engine,'read_lines') else [engine.read_line(v) for v in variants.values()]
        for mode,(value,confidence) in zip(variants,readings):
            candidates.append(FieldCandidate(value,confidence,'padded_'+mode,1.,page,index,name))
    selected=resolve('householder',candidates)
    supporting={c.engine for c in selected.candidates if c.value==selected.value and c.confidence>=.85}
    return candidates if selected.value in values and len(supporting)>=2 else []
