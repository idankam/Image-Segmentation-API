
# Image Segmentation Inference API using FastAPI and PyTorch/ONNX

This project implements an API for performing image segmentation inference using PyTorch and ONNX models, integrated with FastAPI. The API allows inference from both URL-based and file-based image inputs.

## Project Structure

The project consists of several phases, each building upon the previous one:

### Phase 1: Model Conversion and Comparison

- **`phase_1.py`**: 
  - Converts a PyTorch segmentation model to ONNX format.
  - Compares inference results between the original PyTorch model and the ONNX converted model.
  - Visualizes segmentation maps for qualitative comparison.

### Phase 2: Wrapper class - SegmentationModelAI

- **`phase_2.py`**:
  - This module provides classes and utilities to facilitate inference with segmentation models across different platforms and image input types.
  - The main class `SegmentationModelAI` integrates these model platforms through a unified interface, allowing seamless inference on images.
  - Implements unit tests (`phase_2_tests.py`) to validate model inference, ensuring correctness across different input sizes and types.
  - for more information check the detailed docstrings in phase_2.py.

### Phase 3: FastAPI Integration for Inference Service

- **`phase_3.py`**:
  - Implements a FastAPI application for serving segmentation model inferences.
  - Defines endpoints for:
    - Health check (`/healthcheck`).
    - Inference from URL (`/infer/url`).
    - Inference from uploaded file (`/infer/file`).
  - Handles input validation, model selection (PyTorch or ONNX), and result types selection (logits, pixels category distribution, segmentation map).

### Example Client Usage

- **`phase_3_client_example.py`**:
  - Provides example usage of the API endpoints:
    - **URL-based Inference**: Infers segmentation from an image URL.
    - **File-based Inference**: Infers segmentation from an uploaded image file.
  - Demonstrates how to send requests to the API, handle responses, and visualize segmentation maps using `matplotlib`.

## Getting Started

To run the project locally:

1. Clone this repository:
   ```
   git clone https://github.com/idankam/Image-Segmentation-API.git
   cd <repository-directory>
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:
   ```
   uvicorn phase_3:app --reload
   ```

4. Use the example client (`phase_3_client_example.py`) to test the API endpoints:
   ```
   python phase_3_client_exampe.py
   ```

## API Endpoints short documentation (phase 3):

### Health Check

- **Endpoint**: `/healthcheck`
- **Method**: GET
- **Description**: Verify if the server is live.

### URL-based Image Inference

- **Endpoint**: `/infer/url`
- **Method**: POST
- **Parameters**:
  - `infer_model_type`: Type of the segmentation model (`torch` or `onnx`).
  - `requested_results_types`: List of requested results (`logits`, `pixels_distribution`, `segmentation_map`).
  - `url_image`: URL of the image to perform inference on.

### File-based Image Inference

- **Endpoint**: `/infer/file`
- **Method**: POST
- **Parameters**:
  - `infer_model_type`: Type of the segmentation model (`torch` or `onnx`).
  - `requested_results_types`: List of requested results (`logits`, `pixels_distribution`, `segmentation_map`).
  - `file`: Uploaded image file.

## Example Usage

See `phase_3_client_example.py` for example usage of both URL-based and file-based image inference.




## API Endpoints detailed Documentation (phase 3):

This API provides endpoints to perform image segmentation inference using either ONNX or Torch models. The results can be requested in various formats such as logits, pixel category distribution, or segmentation maps.

### Base URL

```
http://<host>:<port>/
```

### Endpoints

#### 1. Health Check Endpoint

**URL:** `/healthcheck`

**Method:** `GET`

**Description:** 
Checks if the server is live and responding.

**Response:**
- `200 OK`
```json
{
    "status": "live"
}
```

#### 2. Infer from URL

**URL:** `/infer/url`

**Method:** `POST`

**Description:** 
Performs image segmentation inference using an image provided via a URL.

**Parameters:**
- `infer_model_type` (Form field, `str`, required): The type of model to use for inference (`onnx` or `torch`).
- `requested_results_types` (Form field, `List`, optional): List of result types to return (`logits`, `pixels_distribution`, `segmentation_map`). If not provided, all result types will be returned.
- `url_image` (Form field, `str`, required): The URL of the image to perform inference on.

**Response:**
- `200 OK`
```json
{
    "Status": "OK",
    "infer_model_type": "<infer_model_type>",
    "image_type": "url",
    "requested_results": {
        "logits": [...],
        "pixels_distribution": [...],
        "segmentation_map": [...]
    }
}
```
- `401 Unauthorized`
```json
{
    "detail": "Fail! Value error: <Error message>"
}
```
- `500 Internal Server Error`
```json
{
    "detail": "Fail! Error: <Error message>"
}
```

#### 3. Infer from File

**URL:** `/infer/file`

**Method:** `POST`

**Description:** 
Performs image segmentation inference using an uploaded image file.

**Parameters:**
- `infer_model_type` (Form field, `str`, required): The type of model to use for inference (`onnx` or `torch`).
- `requested_results_types` (Form field, `List`, optional): List of result types to return (`logits`, `pixels_distribution`, `segmentation_map`). If not provided, all result types will be returned.
- `file` (File, required): The image file to perform inference on.

**Response:**
- `200 OK`
```json
{
    "Status": "OK",
    "infer_model_type": "<infer_model_type>",
    "image_type": "file",
    "requested_results": {
        "logits": [...],
        "pixels_distribution": [...],
        "segmentation_map": [...]
    }
}
```
- `401 Unauthorized`
```json
{
    "detail": "Fail! Value error: <Error message>"
}
```
- `500 Internal Server Error`
```json
{
    "detail": "Fail! Error: <Error message>"
}
```

---

### Error Handling

All endpoints return appropriate HTTP status codes. In case of errors, the response includes a detailed message:

- **401 Unauthorized**: Invalid value in request.
- **500 Internal Server Error**: General server errors, including exceptions during model inference or image processing.

---

### Usage Example

Run the FastAPI server:
   ```
   uvicorn phase_3:app --reload
   ```
run client script (make sure to download requirements first):
   ```
   python phase_3_client_exampe.py
   ```

**python client script example**

```python
    # send image from url:
    data = {
        "infer_model_type": infer_model_type,
        "requested_results_types": requested_results_types,
        "url_image": image_url
    }

    # Send the POST request
    response = requests.post(url, data=data)
```

```python
    # send image file:
    file_path = r"your/image.jpg"
    data = {
        "infer_model_type": infer_model_type,
        "requested_results_types": requested_results_types,
    }
    files = {
        "file": (file_path.split('/')[-1], open(file_path, "rb"), "image/jpeg")
    }

    # Send the POST request
    response = requests.post(url, data=data, files=files)
```
- Check phase_3_client_example.py for more details.


**Alternatives: use curl or postman.**
