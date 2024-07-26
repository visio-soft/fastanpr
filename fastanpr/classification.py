from pathlib import Path
from ultralytics import YOLO
from pydantic import BaseModel
from typing import Union, List

import numpy as np

class Classifier:
    def __init__(self, classification_model: Union[str, Path], device: str):
        self.device = device
        self.model = YOLO(model=classification_model)

    def  run(self, images: List[np.ndarray]):
        results = self.model.predict(images, device=self.device, verbose=False)
        class_names=[]
        for result in results:
            probs = list(result.probs.data)
            classes = result.names

            highest_prob = max(probs)
            highest_prob_index = probs.index(highest_prob)
            class_names.append(classes[highest_prob_index])
        return class_names
