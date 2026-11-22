"""Local pixel evidence for ambiguous address glyphs; no place-name vocabulary."""
from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
import numpy as np
from .ocr import PrecisionRecognizer
if TYPE_CHECKING:
    from .candidates import FieldCandidate


def verify_address(image: np.ndarray,base: str,candidates: list[FieldCandidate],engines: dict[str,PrecisionRecognizer]) -> list[dict]:
    """Constrain observed complete strings using strong isolated glyph readings.

    Whole-line recognizers can prefer the same wrong word. A bounded character
    window supplies different geometric evidence. It never invents a spelling;
    unresolved or conflicting glyphs are ignored and all such decisions remain
    reviewable, because preprocessing variants are not independent sources.
    """
    values={c.value for c in candidates if c.confidence>=.80 and len(c.value)==len(base)
            and sum(a!=b for a,b in zip(base,c.value))<=2}
    indexes=[i for i in range(len(base)) if len({v[i] for v in values})>1]
    if not indexes or len(indexes)>3:return []
    aligned=engines['v5'].read_aligned(image)
    chars=[(ch,x) for group in aligned for ch,x in zip(group['text'],group['positions'])]
    if len(chars)!=len(base):return []
    proof=[]
    for i in indexes:
        lo=round((chars[i-1][1]+chars[i][1])/2) if i else 0
        hi=round((chars[i][1]+chars[i+1][1])/2) if i+1<len(chars) else image.shape[1]
        crop=image[:,max(0,lo):min(image.shape[1],hi)]
        if not crop.size:continue
        evidence=[]
        for name,rec in engines.items():
            views=[crop,cv2.GaussianBlur(crop,(0,0),1.),cv2.GaussianBlur(crop,(0,0),2.)]
            for mode,(value,confidence) in zip(('original','light','smooth'),rec.read_lines(views)):
                if len(value)==1 and value in {v[i] for v in values}:
                    evidence.append(dict(value=value,confidence=confidence,engine=name,mode=mode))
        scores={v:max(c['confidence'] for c in evidence if c['value']==v) for v in {c['value'] for c in evidence}}
        ranked=sorted(scores,key=scores.get,reverse=True)
        if ranked and scores[ranked[0]]>=.80 and (len(ranked)==1 or scores[ranked[1]]<.75) and sum(c['value']==ranked[0] and c['confidence']>=.75 for c in evidence)>=2:
            proof.append(dict(index=i,value=ranked[0],length=len(base),evidence=evidence))
    return proof
