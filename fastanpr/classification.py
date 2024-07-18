from pathlib import Path
from ultralytics import YOLO

from typing import Union, List
import cv2
import numpy as np


class Classifier:
    def __init__(self, device: str ,classification_model: Union[str, Path] = Path(__file__).parent / 'classification_model.pt'):
        self.device = device
        self.clss_model = YOLO(model=classification_model)


    def run(self, images: List[np.ndarray]):
        results = self.clss_model(images, device=self.device, verbose=False,imgsz=224)
        class_names=[]
        for result in results:
            probs = list(result.probs.data)
            classes = result.names

            highest_prob = max(probs)
            highest_prob_index = probs.index(highest_prob)
            # 1 equals to entries/front, 2 equals to exits/back
            class_names.append(1 if classes[highest_prob_index] == 'entries' else 2)
        return class_names

if __name__ == '__main__':
    classifier = Classifier(device='cuda:0')

    img=cv2.imread('filename.jpg')
    names= classifier.run([img])
    print(names)
