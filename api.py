import io
import base64
import uvicorn
import fastanpr
import numpy as np

from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="FastANPR",
    description="A web server for FastANPR hosted using FastAPI",
    version=fastanpr.__version__
)
fast_anpr = fastanpr.FastANPR()


class FastANPRRequest(BaseModel):
    image: str


class FastANPRResponse(BaseModel):
    number_plates: list[fastanpr.NumberPlate] = None


def base64_image_to_ndarray(base64_image_str: str) -> np.ndarray:
    image_data = base64.b64decode(base64_image_str)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image, dtype=np.uint8)


@app.post("/recognise", response_model=FastANPRResponse)
async def recognise(request: FastANPRRequest):
    image = base64_image_to_ndarray(request.image)
    number_plates = (await fast_anpr.run(image))[0]
    return FastANPRResponse(
        number_plates=[fastanpr.NumberPlate.parse_obj(number_plate.__dict__) for number_plate in number_plates]
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
