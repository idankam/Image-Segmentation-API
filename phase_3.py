from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import io
import os

from phase_2 import ONNXModel, SegmentationModelAI, TorchModel


class InferRequest:
    """
        Class to handle the inference request, validate input, set model instance,
        perform inference, and format the results.
    """
    def __init__(self, infer_model_type, requested_results, image_type, image_data) -> None:
        self.infer_model_type = infer_model_type
        self.requested_results = self.validate_requested_results(requested_results)
        self.image_type = image_type
        self.image_data = image_data
        self.model = None
        self.infer_results = None

    @staticmethod
    def validate_requested_results(requested_results):
        """
            Validate and set default requested results if not provided.
        """
        if requested_results is None or len(requested_results) == 0:
            requested_results = ["logits", "pixels_distribution", "segmentation_map"]
        return requested_results

    def set_SegmentationModelAI_instance(self):
        """
            Initialize the SegmentationModelAI instance based on the model type.
        """
        if self.infer_model_type == "onnx":
            onnx_model = ONNXModel.get_default_onnx_converted_segmentation_model()
            self.model = SegmentationModelAI(onnx_model, model_type='onnx', input_size=(512, 512))
        elif self.infer_model_type == "torch":
            try:
                torch_model = TorchModel.get_default_torch_segmentation_model()
                self.model = SegmentationModelAI(torch_model, model_type='torch', input_size=(512, 512))
            except Exception as e:
                print(f"Can't use torch model. Using Onnx instead. error: {e}")
                self.infer_model_type = 'onnx'
                onnx_model = ONNXModel.get_default_onnx_converted_segmentation_model()
                self.model = SegmentationModelAI(onnx_model, model_type='onnx', input_size=(512, 512))
        else:
            raise ValueError(f"Model type '{self.infer_model_type}' is not registered.")

    def __str__(self):
        """
            String representation of the InferRequest object for debugging purposes.
        """
        res = ""
        res += f"infer_model_type: {self.infer_model_type}\n"
        res += f"requested_results: {self.requested_results}\n"
        res += f"image_type: {self.image_type}\n"
        res += f"image_data: {self.image_data}\n"
        res += f"model: {self.model}\n"
        if self.infer_results is None:
            res += f"model: {self.infer_results}\n"
        else:
            res += f"infer_results: probably too long for printing (:\n"
        return res

    def infer(self):
        """
            Perform the inference using the selected model and store the results.
        """
        logits, pixels_distribution, seg_map, _ = self.model(self.image_data)
        results = {'logits': logits.tolist(), 'pixels_distribution': pixels_distribution.tolist(),
                   'segmentation_map': seg_map.tolist()}
        self.infer_results = results

    def get_requested_results(self):
        """
            Extract the requested results from the inference output.
        """
        inference_results = {}
        for result_type in self.requested_results or []:
            if result_type == "logits":
                inference_results['logits'] = self.infer_results['logits']
            elif result_type == "pixels_distribution":
                inference_results['pixels_distribution'] = self.infer_results['pixels_distribution']
            elif result_type == "segmentation_map":
                inference_results['segmentation_map'] = self.infer_results['segmentation_map']
            else:
                inference_results[result_type] = "result type not supported"

        return inference_results


app = FastAPI()


def perform_infer(infer_model_type, requested_results_types, image_type, image_data):
    """
    Perform the inference process and return the results
    """
    request = InferRequest(infer_model_type, requested_results_types, image_type, image_data)
    request.set_SegmentationModelAI_instance()
    request.infer()
    inference_results = request.get_requested_results()

    return {"Status": "OK",
            "infer_model_type": request.infer_model_type,
            "image_type": request.image_type,
            "requested_results": inference_results}


@app.get("/healthcheck")
def health_check():
    """
        Health check endpoint to verify if the server is live.
    """
    return {"status": "live"}


@app.post("/infer/url")
async def infer_from_url(
        infer_model_type: str = Form(...),
        requested_results_types: List = Form(None),
        url_image: str = Form(...),
):
    """
        Endpoint to perform inference from an image URL.
    """
    try:
        return perform_infer(infer_model_type, requested_results_types, 'url', url_image)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Fail! Value error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fail! Error: {str(e)}")


@app.post("/infer/file")
async def infer_from_file(
        infer_model_type: str = Form(...),
        requested_results_types: List = Form(None),
        file: UploadFile = File(...),
):
    """
    Endpoint to perform inference from an uploaded image file.
    """
    try:
        image_content = await file.read()
        return perform_infer(infer_model_type, requested_results_types, 'file', image_content)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Fail! Value error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fail! Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("running")
    uvicorn.run(app, host="0.0.0.0", port=8000)
