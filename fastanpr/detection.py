from pathlib import Path
from ultralytics import YOLO
from pydantic import BaseModel
from typing import Union, List

import numpy as np


class Detection(BaseModel):
    image: np.ndarray
    box: List[int]
    conf: float
    gate_pos: str

    class Config:
        frozen = True
        arbitrary_types_allowed = True


class Detector:
    def __init__(self, detection_model: Union[str, Path], device: str):
        self.device = device
        self.model = YOLO(model=detection_model)

    def run(self, images: List[np.ndarray]) -> List[List[Detection]]:
        predictions = self.model.predict(images, device=self.device, verbose=False)
        results = []
        for image, detection in zip(images, predictions):
            image_detections = []
            if detection.boxes:
                det_boxes = detection.boxes.cpu().data.numpy().astype(int).tolist()
                det_confs = detection.boxes.cpu().conf.numpy().tolist()
                det_classes = detection.boxes.cpu().cls.tolist()
                for det_box, det_conf, det_class in zip(det_boxes, det_confs, det_classes):
                    det_box[0], det_box[2], det_box[1], det_box[3] = det_box[0] - 15, det_box[2] + 15, det_box[1] - 15, det_box[3] + 15
                    det_box[:4]=self.clip(det_box[:4],0,max(image.shape[1],image.shape[0]))

                    x_min, x_max, y_min, y_max = det_box[0], det_box[2], det_box[1], det_box[3]
                    image_detections.append(
                        Detection(image=image[y_min:y_max, x_min:x_max, :], box=det_box[:4], conf=det_conf, gate_pos=detection.names[int(det_class)])
                    )
            results.append(image_detections)
        return results
    def clip(self,values, min_value, max_value):
        """
        Clip the given integer value to the specified range [min_value, max_value].

        Parameters:
        value (int): The integer value to be clipped.
        min_value (int): The minimum allowed value.
        max_value (int): The maximum allowed value.

        Returns:
        int: The clipped value.
        """
        lst=[]
        for v in values:
            lst.append(max(min_value, min(v, max_value)))
        return lst
