"""
Segmentation Model AI

This module provides classes and utilities to facilitate inference with segmentation models across different platforms:
- PyTorch models (`TorchModel`)
- ONNX models (`ONNXModel`)
- TensorFlow models (`TensorFlowModel`)

The main class `SegmentationModelAI` integrates these model platforms through a unified interface, allowing seamless inference on images.

Classes:
--------
- ImageProcessor: Utility class to handle image preprocessing and validation.
- BaseModel (abstract class): Defines an interface for model inference.
- ModelRegistry: Registry to manage different model types and their creation.
- TorchModel: Implementation of BaseModel for PyTorch models.
- ONNXModel: Implementation of BaseModel for ONNX models.
- TensorFlowModel: Implementation of BaseModel for TensorFlow models.
- SegmentationModelAI: High-level interface to load a model of a specified type and perform inference on images.

Usage:
------
Instantiate SegmentationModelAI with a model and its type (torch, onnx, tensorflow), and call it with an image input to get segmentation results.

Example:
--------
if __name__ == "__main__":
    # Replace with your model paths or URLs
    torch_model_path = 'path_to_your_torch_model.pth'
    onnx_model_path = 'path_to_your_onnx_model.onnx'
    tf_model_path = 'path_to_your_tf_model'

    # Example using PyTorch model
    pytorch_model = torch.load(torch_model_path)
    segmentation_model = SegmentationModelAI(pytorch_model, 'torch')
    try:
        result = segmentation_model('path_to_your_image.jpg')
        print(result)
    except Exception as e:
        print(f"Error during inference: {e}")
"""

import torch
import onnxruntime as ort
from torchvision import transforms, models
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from abc import ABC, abstractmethod
import tensorflow as tf

class ImageProcessor:
    """
    Utility class for image preprocessing and validation.

    Provides methods to load images from different sources (path, URL, bytes, PIL Image, torch Tensor),
    preprocess them by resizing, normalizing, and converting to tensors, and validate image dimensions
    and formats.
    """

    @staticmethod
    def preprocess_image(image, size=(512, 512)):
        """
        Preprocess the input image by resizing, normalizing, and converting to tensor.
        """
        image = ImageProcessor.load_image(image)
        ImageProcessor.validate_image(image)
        transform = ImageProcessor.get_transform(size)
        return transform(image).unsqueeze(0)

    @staticmethod
    def load_image(image):
        """
        Load the image from different formats (path, URL, bytes, PIL Image, torch Tensor).
        """
        if isinstance(image, str):
            return ImageProcessor.load_image_from_path_or_url(image)
        elif isinstance(image, bytes):
            return ImageProcessor.load_image_from_bytes(image)
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        elif isinstance(image, torch.Tensor):
            return image  # Assuming tensor is already preprocessed
        else:
            raise TypeError(
                "Unsupported image type. Expected str (file path or URL), bytes, PIL.Image.Image, or torch.Tensor."
            )

    @staticmethod
    def load_image_from_path_or_url(image_path_or_url):
        """
        Load the image from a file path or URL.
        """
        try:
            if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
                response = requests.get(image_path_or_url)
                response.raise_for_status()  # Raise an HTTPError for bad responses
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_path_or_url).convert('RGB')
            return image
        except Exception as e:
            raise RuntimeError(f"Error loading image from path or URL: {e}")

    @staticmethod
    def load_image_from_bytes(image_bytes):
        """
        Load the image from bytes.
        """
        try:
            return Image.open(BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image from bytes: {e}")

    @staticmethod
    def get_transform(size):
        """
        Get the transformation pipeline.
        """
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def validate_image(image):
        """
        Validate the loaded image for expected dimensions and format.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Input is not a valid PIL Image.")
        if image.mode != 'RGB':
            image = image.convert('RGB')  # Convert to RGB if not already
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError("Image dimensions are invalid.")
        # Add additional validation as needed

    @staticmethod
    def validate_tensor(tensor):
        """
        Validate the input tensor meets expected criteria.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input is not a valid PyTorch tensor.")
        if tensor.ndim != 4:
            raise ValueError("Input tensor dimensions are invalid. Expected 4D tensor.")

        # Add additional validation for tensor shape, dtype, etc.

        return tensor


class BaseModel(ABC):
    """
    Abstract base class for model inference.

    Defines an interface for model inference and provides a structure for different model implementations.
    """

    def __init__(self, model, input_size=(512, 512)):
        self.model = model
        self.input_size = input_size

    @abstractmethod
    def infer(self, input_tensor):
        """
        Run inference on the input tensor and return the output.
        """
        pass


class ModelRegistry:
    """
    Registry for managing different model types and their creation.

    Provides registration of model types and creation of model instances based on the specified type.
    """

    _registry = {}

    @classmethod
    def register(cls, name):
        """
        Decorator to register a model type.
        """
        def decorator(model_class):
            cls._registry[name] = model_class
            return model_class

        return decorator

    @classmethod
    def create_model(cls, model, model_type, input_size=(512, 512)):
        """
        Create a model instance based on the specified type.
        """
        if model_type not in cls._registry:
            raise ValueError(f"Model type '{model_type}' is not registered.")

        return cls._registry[model_type](model, input_size)


@ModelRegistry.register('torch')
class TorchModel(BaseModel):
    """
    Implementation of BaseModel for PyTorch models.

    Handles loading of PyTorch models and running inference.
    """

    def __init__(self, model, input_size=(512, 512)):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Provided model is not a valid PyTorch model.")
        super().__init__(model, input_size)

    def infer(self, input_tensor):
        """
        Run inference using the PyTorch model.
        """
        try:
            input_tensor = ImageProcessor.validate_tensor(input_tensor)
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)
            return output
        except Exception as e:
            raise RuntimeError(f"Error during PyTorch model inference: {e}")


@ModelRegistry.register('onnx')
class ONNXModel(BaseModel):
    """
    Implementation of BaseModel for ONNX models.

    Handles loading of ONNX models and running inference.
    """

    def __init__(self, model, input_size=(512, 512)):
        if isinstance(model, str):
            try:
                model = ort.InferenceSession(model)
            except (IOError, ort.OrtInvalidModel, ort.OrtInvalidArgument) as e:
                raise ValueError(f"Failed to load ONNX model: {e}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error loading ONNX model: {e}")
        elif not isinstance(model, ort.InferenceSession):
            raise TypeError("Provided model is not a valid ONNX InferenceSession or path to an ONNX model file.")
        super().__init__(model, input_size)

    def infer(self, input_tensor):
        """
        Run inference using the ONNX model.
        """
        try:
            input_tensor = ImageProcessor.validate_tensor(input_tensor)
            ort_inputs = {self.model.get_inputs()[0].name: input_tensor.numpy()}
            ort_outputs = self.model.run(None, ort_inputs)
            return ort_outputs[0]
        except Exception as e:
            raise RuntimeError(f"Error during ONNX model inference: {e}")


@ModelRegistry.register('tensorflow')
class TensorFlowModel(BaseModel):
    """
    Implementation of BaseModel for TensorFlow models.

    Handles loading of TensorFlow models and running inference.
    """

    def __init__(self, model, input_size=(512, 512)):
        if isinstance(model, str):
            try:
                model = tf.saved_model.load(model)
            except (OSError, tf.errors.NotFoundError, ValueError) as e:
                raise ValueError(f"Failed to load TensorFlow model: {e}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error loading TensorFlow model: {e}")
        elif not isinstance(model, tf.Module):
            raise TypeError("Provided model is not a valid TensorFlow model or path to a TensorFlow model directory.")
        super().__init__(model, input_size)

    def infer(self, input_tensor):
        """
        Run inference using the TensorFlow model.
        """
        try:
            input_tensor = ImageProcessor.validate_tensor(input_tensor)
            input_array = input_tensor.numpy()
            input_array = tf.convert_to_tensor(input_array)
            outputs = self.model(input_array, training=False)
            return outputs[0].numpy()  # Assuming single output
        except Exception as e:
            raise RuntimeError(f"Error during TensorFlow model inference: {e}")


class SegmentationModelAI:
    """
    High-level interface to facilitate inference with segmentation models.

    Integrates different model platforms (PyTorch, ONNX, TensorFlow) through a unified interface.
    """

    def __init__(self, model, model_type, input_size=(512, 512)):
        """
        Initialize the SegmentationModelAI instance with a model and its type.

        Parameters:
        -----------
        model : str or object
            Path to the model file or model object itself.
        model_type : str
            Type of the model ('torch', 'onnx', 'tensorflow').
        input_size : tuple, optional
            Input size of the image (default is (512, 512)).

        Raises:
        -------
        ValueError if the model type is not registered.
        """
        self.model = ModelRegistry.create_model(model, model_type, input_size)
        self.input_size = input_size

    def __call__(self, image):
        """
        Perform inference on the given image.

        Parameters:
        -----------
        image : str or bytes or PIL.Image.Image or torch.Tensor
            Input image to perform segmentation on.

        Returns:
        --------
        Output segmentation results.

        Raises:
        -------
        TypeError if the image input type is unsupported.
        RuntimeError if an error occurs during image preprocessing or model inference.
        """
        try:
            image = ImageProcessor.load_image(image)
            ImageProcessor.validate_image(image)
            input_tensor = ImageProcessor.preprocess_image(image, self.input_size)
            return self.model.infer(input_tensor)
        except Exception as e:
            raise RuntimeError(f"Error during inference: {e}")





def get_torch_segmentation_model():
    weights = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT  # Get the most up-to-date weights
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
    model.eval()  # Set model to evaluation mode
    return model


def main():
    torch_model = get_torch_segmentation_model()
    onnx_model_path = 'deeplabv3_mobilenet_v3_large.onnx'
    onnx_model = ort.InferenceSession(onnx_model_path)

    # # Using Torch model
    # model = SegmentationModelAI(torch_model, model_type='torch', input_size=(512, 512))
    # result = model('path/to/image.jpg')  # Local image
    # result = model('http://example.com/image.jpg')  # Image URL
    # result = model(image_bytes)  # Image bytes
    #
    # # Using ONNX model
    # model = SegmentationModelAI(onnx_model, model_type='onnx', input_size=(512, 512))
    # result = model('path/to/image.jpg')  # Local image
    # result = model('http://example.com/image.jpg')  # Image URL
    # result = model(image_bytes)  # Image bytes


if __name__ == '__main__':
    main()
