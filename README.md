

## Phase 3 - FastAPI Inference Documentation

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
    "url": "<url_image>",
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
    "file": "<file_name>",
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

**python client script**
- Check phase_3_client_example.py for more details. Should be used from IDE for visualize.

- Alternatives: use curl or postman.
