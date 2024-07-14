

## Phase 3 - FastAPI Inference Documentation

This API provides endpoints to perform image segmentation inference using either ONNX or Torch models. The results can be requested in various formats such as logits, pixel distribution, or segmentation maps.

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
- `500 Internal Server Error`
```json
{
    "detail": "Error message"
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
    "detail": "Fail! Value error: Error message"
}
```
- `500 Internal Server Error`
```json
{
    "detail": "Fail! Error: Error message"
}
```

---

### Error Handling

All endpoints return appropriate HTTP status codes. In case of errors, the response includes a detailed message:

- **401 Unauthorized**: Invalid value in request.
- **500 Internal Server Error**: General server errors, including exceptions during model inference or image processing.

---

### Usage Example

**Infer from URL Example:**

```bash
curl -X POST "http://localhost:8000/infer/url" \
-H "accept: application/json" \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "infer_model_type=onnx&url_image=https://example.com/image.jpg&requested_results_types=logits&requested_results_types=pixels_distribution"
```

**Infer from File Example:**

```bash
curl -X POST "http://localhost:8000/infer/file" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "infer_model_type=torch" \
-F "requested_results_types=logits" \
-F "requested_results_types=pixels_distribution" \
-F "file=@/path/to/image.jpg"
```

This documentation provides an overview of the available endpoints, their parameters, and example usage to help you interact with the API effectively.
