import argparse
import os

import numpy as np
from PIL import Image, ImageFile
from pydantic import BaseModel
from tqdm import tqdm
from typing import Union, List
import cv2

from .yolo_rt.models.cudart_api import TRTEngine
from .yolo_rt.models.utils import blob, letterbox,det_postprocess

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Detection(BaseModel):
    image: np.ndarray
    box: List[int]
    conf: float

    class Config:
        frozen = True
        arbitrary_types_allowed = True


class Detect_RT():
    def __init__(self, model_path):

        self.Engine = TRTEngine(model_path)
        self.H, self.W = self.Engine.inp_info[0].shape[-2:]
        self.valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']


    def default_det(self, rgbs):
        results = []
        for rgb in rgbs:
            image_detections = []

            rgb_temp, ratio, dwdh = letterbox(np.array(rgb), (self.W, self.H))
            tensor = blob(rgb_temp, return_seg=False)
            dwdh = np.array(dwdh * 2, dtype=np.float32)
            tensor = np.ascontiguousarray(tensor)
            # inference
            data = self.Engine(tensor)

            bboxes, scores, labels = det_postprocess(data)
            if bboxes.size != 0:
                bboxes -= dwdh
                bboxes /= ratio

                for (bbox, score, label) in zip(bboxes, scores, labels):
                    bbox = bbox.round().astype(np.int32).tolist()
                    bbox = list(map(lambda x: max(0, x), bbox))
                    x1, y1, x2, y2 = bbox
                    if (x2 >x1) and (y2 > y1):
                        image_detections.append(
                            Detection(image= rgb[y1:y2, x1:x2,:], box=bbox, conf=score)
                        )
            results.append(image_detections)
        return results
    def run(self, image_list:np.ndarray):
        result = self.default_det(image_list)
        return result

    def search_files(self,image_path):
        image_path_list=os.listdir(image_path)
        image_list = [cv2.cvtColor(cv2.imread(os.path.join(image_path, file)), cv2.COLOR_BGR2RGB) for file in image_path_list]
        return  image_list

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 TensorRT Object Detection')

    parser.add_argument('--model_path', type=str, default='YOLOv8_TensorRT/24_nov_weights_w_one_batch/best.engine',
                        help='Path to the TensorRT engine file')
    parser.add_argument('--image_path', type=str, help='Path to the directory containing test images', default=None)


    args = parser.parse_args()
    detector = Detect_RT(args.model_path)
    image_list = detector.search_files(args.image_path)
    detector.run(image_list=image_list)


if __name__ == '__main__':
    import time
    s=time.perf_counter()
    main()
    print(f'{time.perf_counter()-s}')
