from typing import List
from pydantic import BaseModel


class NumberPlate(BaseModel):
    det_box: List[int]
    det_conf: float
    rec_poly: List[List[int]] = None
    rec_text: str = None
    rec_conf: float = None

    class Config:
        frozen = True
