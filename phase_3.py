from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import io
import os

from SegmentationModelAI import ONNXModel, SegmentationModelAI, TorchModel


class InferRequest:
    def __init__(self, infer_model_type, requested_results, image_type, image_data) -> None:
        self.infer_model_type = infer_model_type
        self.requested_results = self.validate_requested_results(requested_results)
        self.image_type = image_type
        self.image_data = image_data
        self.model = None
        self.infer_results = None

    @staticmethod
    def validate_requested_results(requested_results):
        # if not specified - return all result types
        if requested_results is None or len(requested_results) == 0:
            requested_results = ["logits", "pixels_distribution", "segmentation_map"]
        return requested_results

    def set_SegmentationModelAI_instance(self):
        if self.infer_model_type == "onnx":
            onnx_model = ONNXModel.get_default_onnx_converted_segmentation_model()
            self.model = SegmentationModelAI(onnx_model, model_type='onnx', input_size=(512, 512))
        elif self.infer_model_type == "torch":
            try:
                torch_model = TorchModel.get_default_torch_segmentation_model()
                self.model = SegmentationModelAI(torch_model, model_type='torch', input_size=(512, 512))
            except Exception as e:
                print(f"Can't use torch model. Using Onnx instead. error: {e}")
                onnx_model = ONNXModel.get_default_onnx_converted_segmentation_model()
                self.model = SegmentationModelAI(onnx_model, model_type='onnx', input_size=(512, 512))
        else:
            raise ValueError(f"Model type '{self.infer_model_type}' is not registered.")

    def __str__(self):
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
        print("infer")
        logits, pixels_distribution, seg_map, _ = self.model(self.image_data)
        results = {'logits': logits.tolist(), 'pixels_distribution': pixels_distribution.tolist(),
                   'segmentation_map': seg_map.tolist()}
        self.infer_results = results
        print(results)

    def get_requested_results(self):

        # Prepare the results based on the requested results
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


def infer(infer_model_type, image):
    onnx_model = ONNXModel.get_default_onnx_converted_segmentation_model()
    segmentation_model_onnx = SegmentationModelAI(onnx_model, model_type='onnx', input_size=(512, 512))
    torch_model = TorchModel.get_default_torch_segmentation_model()
    segmentation_model_torch = SegmentationModelAI(torch_model, model_type='torch', input_size=(512, 512))

    if infer_model_type == "onnx":
        logits, pixels_distribution, seg_map, _ = segmentation_model_onnx(image)
        return logits.tolist(), pixels_distribution.tolist(), seg_map.tolist()
    elif infer_model_type == "torch":
        logits, pixels_distribution, seg_map, _ = segmentation_model_torch(image)
        return logits.tolist(), pixels_distribution.tolist(), seg_map.tolist()
    else:
        raise ValueError(f"Model type '{infer_model_type}' is not registered.")


def get_requested_results(requested_results, logits, pixels_distribution, seg_map):
    # if not specified - return all result types
    if requested_results is None or len(requested_results) == 0:
        requested_results = ["logits", "pixels_distribution", "segmentation_map"]

    # Prepare the results based on the requested results
    inference_results = {}
    for result_type in requested_results or []:
        if result_type == "logits":
            inference_results['logits'] = logits
        elif result_type == "pixels_distribution":
            inference_results['pixels_distribution'] = pixels_distribution
        elif result_type == "segmentation_map":
            inference_results['segmentation_map'] = seg_map
        else:
            inference_results[result_type] = "result type not supported"

    return inference_results


def perform_infer(infer_model_type, requested_results_types, image_type, image_data):
    request = InferRequest(infer_model_type, requested_results_types, image_type, image_data)
    request.set_SegmentationModelAI_instance()
    request.infer()
    inference_results = request.get_requested_results()

    return {"infer_model_type": infer_model_type,
            image_type: image_data,
            "requested_results": inference_results}


@app.get("/")
def read_root():
    current_dir = os.getcwd()
    return {"current_directory": current_dir}


@app.get("/healthcheck")
def health_check():
    return {"status": "live"}


@app.post("/infer/url")
async def infer_from_url(
        infer_model_type: str = Form(...),
        requested_results_types: List = Form(None),
        url_image: str = Form(...),
):
    try:
        return perform_infer(infer_model_type, requested_results_types, 'url', url_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer/file")
async def infer_from_file(
        infer_model_type: str = Form(...),
        requested_results_types: List = Form(None),
        file: UploadFile = File(...),
):
    try:
        # Read image file asynchronously
        image_content = await file.read()

        # # Perform inference
        # # logits, pixels_distribution, seg_map = infer(infer_model_type, contents)
        # # inference_results = get_requested_results(requested_results, logits, pixels_distribution, seg_map)
        #
        # request = InferRequest(infer_model_type, requested_results_types, 'file', image_content)
        # request.infer()
        # inference_results = request.get_requested_results()
        #
        # # # Save the image (for demonstration)
        # # image.save(f"uploaded_{file.filename}")
        #
        # # Return details
        # return {
        #     "infer_model_type": infer_model_type,
        #     "filename": file.filename,
        #     "requested_results": inference_results,
        # }
        return perform_infer(infer_model_type, requested_results_types, 'file', image_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("running")
    uvicorn.run(app, host="0.0.0.0", port=8000)
