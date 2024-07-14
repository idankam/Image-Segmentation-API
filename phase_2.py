"""
PHASE 2

This module provides classes and utilities to facilitate inference with segmentation models across different platforms:
- PyTorch models (`TorchModel`)
- ONNX models (`ONNXModel`)

The main class `SegmentationModelAI` integrates these model platforms through a unified interface,
allowing seamless inference on images.

Classes:
--------
- ImageLoaderRegistry: Registry for managing different image loading methods based on data types.
- ImageProcessor: Utility class to handle image preprocessing and validation.
- BaseModel (abstract class): Defines an interface for model inference.
- ModelRegistry: Registry to manage different model types and their creation.
- TorchModel: Implementation of BaseModel for PyTorch models.
- ONNXModel: Implementation of BaseModel for ONNX models.
- SegmentationModelAI: High-level interface to load a model of a specified type and perform inference on images.

Usage:
------
Instantiate SegmentationModelAI with a model and its type (torch, onnx),
and call it with an image input to get segmentation results.

Example:
--------
if __name__ == "__main__":
    from SegmentationModelAI import SegmentationModelAI, TorchModel, ONNXModel

    # Example using a PyTorch model
    torch_model = TorchModel.get_default_torch_segmentation_model()
    segmentation_model_torch = SegmentationModelAI(torch_model, model_type='torch', input_size=(512, 512))
    try:
        logits, pixels_distribution, seg_map, seg_map_image = segmentation_model_torch('path_to_your_image.jpg')
        print(logits)
        seg_map_image.save("file_name.jpg")
    except Exception as e:
        print(f"Error during PyTorch model inference: {e}")

    # Example using an ONNX model
    onnx_model = ONNXModel.get_default_onnx_converted_segmentation_model()
    segmentation_model_onnx = SegmentationModelAI(onnx_model, model_type='onnx', input_size=(512, 512))
    try:
        logits, pixels_distribution, seg_map, seg_map_image = segmentation_model_onnx('path_to_your_image.jpg')
        print(logits)
        seg_map_image.save("file_name.jpg")
    except Exception as e:
        print(f"Error during ONNX model inference: {e}")
"""
from typing import Tuple, Union, Dict, Type, Callable

import matplotlib.pyplot as plt
import torch
import onnxruntime as ort
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import requests
from io import BytesIO
from abc import ABC, abstractmethod
import numpy as np
from torchvision.models.segmentation import deeplabv3

from phase_1 import get_logits, ModelsTypes, get_pixel_category_distribution, get_segmentation_map, PHASE_1_ONNX_MODEL_PATH

DEFAULT_IMAGE_SIZE = (512, 512)


# Separate class for managing image loading methods registration
class ImageLoaderRegistry:

    """
    ImageLoaderRegistry: Registry for managing different image loading methods based on data types.

    Provides registration and retrieval of image loading methods
    for various data types (str, bytes, PIL.Image.Image, torch.Tensor).
    """
    _loaders_registry: Dict[Type, Callable] = {}

    @classmethod
    def register_loader(cls, data_type: Type) -> Callable:
        """
        Decorator to register a new image loader for a specific data type.
        """

        def decorator(loader_function: Callable) -> None:
            cls._loaders_registry[data_type] = loader_function

        return decorator

    @classmethod
    def check_if_data_type_registered(cls, data_type: Type) -> bool:
        """
        Check if a data type is already registered for image loading.
        """
        if data_type in ImageLoaderRegistry._loaders_registry:
            return True
        return False

    @classmethod
    def get_loader_func(cls, data_type: Type) -> Callable:
        """
        Get the loader function for a specific data type.
        """
        return cls._loaders_registry[data_type]


class ImageProcessor:
    """
    ImageProcessor: Utility class for image preprocessing and validation.

    This class provides methods for loading images from various sources (file path, URL, bytes, PIL Image, torch Tensor),
    preprocessing them by resizing, normalizing, and converting to tensors, and validating image dimensions and formats.

    To support additional image formats:
        - Add new 'load_image_from...' methods for the specific formats.
        - Modify the 'load_image' method to include these new formats if necessary.
    """

    @staticmethod
    def preprocess_image(image: Union[str, bytes, Image.Image, torch.Tensor],
                         size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
        """
        Preprocess the input image by resizing, normalizing, and converting to tensor.

        Parameters:
        -----------
        image : str or bytes or PIL.Image.Image or torch.Tensor
            Input image to preprocess.
        size : tuple
            Target size (height, width) to resize the image.

        Returns:
        --------
        torch.Tensor
            Preprocessed image tensor ready for model input.

        Raises:
        ------
        ValueError
            If the image size dimensions are invalid.
        """

        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("Image size dimensions must be greater than zero.")

        loaded_image = ImageProcessor.load_image(image)
        if isinstance(loaded_image, torch.Tensor):
            return loaded_image
        else:
            ImageProcessor.validate_image(loaded_image)
            transform = ImageProcessor.get_transform(size)
            return transform(loaded_image).unsqueeze(0)

    @staticmethod
    def load_image(image: Union[str, bytes, Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        """
        Load the image from different formats (path, URL, bytes, PIL Image, torch Tensor).

        Parameters:
        -----------
        image : str or bytes or PIL.Image.Image or torch.Tensor
            Input image to load.

        Returns:
        --------
        PIL.Image.Image or torch.Tensor
            Loaded image in PIL.Image.Image format or torch.Tensor format.

        Raises:
        ------
        TypeError
            If the input image type is unsupported.
        """
        image_type = type(image)
        if ImageLoaderRegistry.check_if_data_type_registered(image_type):
            load_image_from_func = ImageLoaderRegistry.get_loader_func(image_type)
            return load_image_from_func(image)
        else:
            raise TypeError(
                "Unsupported image type. Expected str (file path or URL), bytes, PIL.Image.Image, or torch.Tensor."
            )

    @staticmethod
    @ImageLoaderRegistry.register_loader(str)
    def load_image_from_path_or_url(image_path_or_url: str) -> Image.Image:
        """
        Load the image from a file path or URL.

        Parameters:
        -----------
        image_path_or_url : str
            Local file path or URL to load the image from.

        Returns:
        --------
        PIL.Image.Image
            Loaded image in PIL.Image.Image format.

        Raises:
        ------
        RuntimeError
            If an error occurs during image loading from path or URL.
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
    @ImageLoaderRegistry.register_loader(bytes)
    def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
        """
        Load the image from bytes.

        Parameters:
        -----------
        image_bytes : bytes
            Input image bytes to load.

        Returns:
        --------
        PIL.Image.Image
            Loaded image in PIL.Image.Image format.

        Raises:
        ------
        RuntimeError
            If an error occurs during image loading from bytes.
        """
        try:
            return Image.open(BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image from bytes: {e}")

    @staticmethod
    @ImageLoaderRegistry.register_loader(Image.Image)
    def load_image_from_pil(image_pil: Image.Image) -> Image.Image:
        """
        Load the image from a PIL Image (in this case, just make sure image is in RGB).

        Parameters:
        -----------
        image_pil : PIL.Image.Image
            PIL Image to load.

        Returns:
        --------
        PIL.Image.Image
            Loaded image in PIL.Image.Image format.
        """
        return image_pil.convert('RGB')

    @staticmethod
    @ImageLoaderRegistry.register_loader(torch.Tensor)
    def load_image_from_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Load the image from a torch Tensor.

        Parameters:
        -----------
        image_tensor : torch.Tensor
            Input image tensor to load.

        Returns:
        --------
        torch.Tensor
            Loaded image tensor in appropriate shape (batched) - (1,3,_,_).

        Raises:
        ------
        ValueError
            If the input tensor dimensions are invalid.
        """
        if image_tensor.ndim == 4:
            return image_tensor  # Assuming tensor is already preprocessed and with correct dimensions (batched)
        elif image_tensor.ndim == 3:
            return image_tensor.unsqueeze(0)  # Assuming tensor is already preprocessed
        else:
            raise ValueError("Image (tensor type) dimensions should be 3 (single input) or 4 (already batched).")

    @staticmethod
    def get_transform(size: Tuple[int, int]) -> transforms.Compose:
        """
        Get the transformation pipeline.
        Note: Normalize using mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        is a common practice in various vision tasks and models. those parameters derived from ImageNet dataset stats.

        Parameters:
        -----------
        size : tuple
            Target size (height, width) to resize the image.

        Returns:
        --------
        transforms.Compose
            Transformation pipeline for image preprocessing.
        """
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def validate_image(image: Image.Image) -> None:
        """
        Validate the image dimensions and format.

        Parameters:
        -----------
        image : PIL.Image.Image
            Input image to validate.

        Raises:
        ------
        ValueError
            If the image size dimensions are invalid.
        """
        if isinstance(image, torch.Tensor):
            return  # Assuming tensor is already preprocessed
        if not isinstance(image, Image.Image):
            raise TypeError("Input is not a valid PIL Image.")
        if image.mode != 'RGB':
            image = image.convert('RGB')  # Convert to RGB if not already
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError("Image dimensions must be greater than zero.")

    @staticmethod
    def validate_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        Validate the input tensor meets expected criteria.

        Parameters:
        -----------
        tensor : torch.Tensor
            Input tensor to validate.

        Returns:
        --------
        torch.Tensor
            The input tensor if validation is successful.

        Raises:
        ------
        TypeError
            If the input is not a valid PyTorch tensor.
        ValueError
            If the input tensor dimensions are invalid (expected 4D tensor).
        """

        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input is not a valid PyTorch tensor.")
        if tensor.ndim != 4:
            raise ValueError("Input tensor dimensions are invalid. Expected 4D tensor.")

        return tensor

    @staticmethod
    def image_to_bytes(image_local_path: str) -> bytes:
        """
        Convert a local image file to bytes.

        Parameters:
        -----------
        image_local_path : str
            Path to the local image file.

        Returns:
        --------
        bytes
            Byte representation of the image file in JPEG format.
        """
        with Image.open(image_local_path) as img:
            img = img.convert('RGB')  # Ensure the image is in RGB format
            byte_arr = BytesIO()
            img.save(byte_arr, format='JPEG')  # Save the image to the BytesIO object in JPEG format
            return byte_arr.getvalue()


class BaseModel(ABC):
    """
    BaseModel: Abstract base class for segmentation models.

    Defines the interface for model inference, provides a structure for different model implementations
    and utility methods for loading and running models.

    To extend SegmentationModelAI for supporting more model types:
        - Implement a new class inheriting from 'BaseModel'.
        - Register the new model class with 'ModelRegistry' using '@ModelRegistry.register('model_type')'.
        - Implement the 'infer' method to define model-specific inference logic.
        - Add model-specific initialization and configuration in '__init__'.
        - Ensure error handling for model loading and inference errors.
    """

    def __init__(self, model, input_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> None:
        self.model = model
        self.input_size = input_size

    @abstractmethod
    def infer(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Image.Image]:
        """
        Run inference on the input tensor and return the output (logits, pixels_distribution, seg_map, seg_map_image).
        """
        pass


class ModelRegistry:
    """
    ModelRegistry: Registry for managing different model types and their creation.

    Provides registration of model types ('torch', 'onnx', and more if implemented)
    and creation of model instances based on the specified type.
    """

    _models_registry = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """
        Decorator to register a model type.
        """

        def decorator(model_class: callable) -> callable:
            cls._models_registry[name] = model_class
            return model_class

        return decorator

    @classmethod
    def create_model(cls, model: Union[str, object], model_type: str,
                     input_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> BaseModel:
        """
        Create a model instance based on the specified type.
        """
        if model_type not in cls._models_registry:
            raise ValueError(f"Model type '{model_type}' is not registered.")

        return cls._models_registry[model_type](model, input_size)


@ModelRegistry.register('torch')
class TorchModel(BaseModel):
    """
    TorchModel: Implementation of BaseModel for PyTorch models.

    Handles loading of PyTorch models and running inference.
    """

    def __init__(self, model: torch.nn.Module, input_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Provided model is not a valid PyTorch model. model type is: {type(model)}")
        if not isinstance(model, deeplabv3.DeepLabV3):
            raise TypeError(f"Provided model is not a valid segmentation model. model type is: {type(model)}")
        super().__init__(model, input_size)

    def infer(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Image.Image]:
        """
        Run inference using the PyTorch model.

        Parameters:
        input_tensor (torch.Tensor): The preprocessed input tensor to be fed into the PyTorch model.
                                     The tensor should be validated before being used for inference.

        Returns:
        tuple:
            - logits (numpy.ndarray): The raw output logits from the PyTorch model.
            - pixels_distribution (numpy.ndarray): The distribution of pixel categories derived from the logits.
            - seg_map (numpy.ndarray): The segmentation map created from the logits, representing the
                                      predicted class for each pixel.
            - seg_map_image (PIL.Image.Image): The segmentation map converted to an image format for
                                               visualization.

        Raises:
        RuntimeError: If an error occurs during the inference process.
        """

        input_tensor = ImageProcessor.validate_tensor(input_tensor)

        # Check input tensor size
        if tuple(input_tensor.shape[-2:]) != self.input_size:
            raise ValueError(
                f"Input tensor size {tuple(input_tensor.shape[-2:])} does not match expected input size {self.input_size} of this model.")

        try:
            self.model.eval()
            with torch.no_grad():
                output = self.model(input_tensor)

            logits = get_logits(output, ModelsTypes.TORCH)
            pixels_distribution = get_pixel_category_distribution(logits)
            seg_map_image, seg_map = get_segmentation_map(logits, self.input_size)

            return logits, pixels_distribution, seg_map, seg_map_image
        except Exception as e:
            raise RuntimeError(f"Error during PyTorch model inference: {e}")

    @staticmethod
    def get_default_torch_segmentation_model() -> torch.nn.Module:
        weights = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT  # Get the most up-to-date weights
        model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
        model.eval()  # Set model to evaluation mode
        return model


@ModelRegistry.register('onnx')
class ONNXModel(BaseModel):
    """
    ONNXModel: Implementation of BaseModel for ONNX models.

    Handles loading of ONNX models and running inference.
    """

    def __init__(self, model: str or Path or ort.InferenceSession,
                 input_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> None:
        if isinstance(model, str) or isinstance(model, Path):
            try:
                model = ort.InferenceSession(model)
            except Exception as e:
                raise RuntimeError(f"Unexpected error loading ONNX model: {e}")
        elif not isinstance(model, ort.InferenceSession):
            raise TypeError(f"Provided model is not a valid ONNX InferenceSession or path to an ONNX model file. "
                            f"model type is: {type(model)}")
        super().__init__(model, input_size)

    def infer(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Image.Image]:
        """
        Run inference using the ONNX model and return segmentation map as output.
        """

        input_tensor = ImageProcessor.validate_tensor(input_tensor)

        # Check input tensor size
        if tuple(input_tensor.shape[-2:]) != self.input_size:
            raise ValueError(
                f"Input tensor size {tuple(input_tensor.shape[-2:])} does not match"
                f" expected input size {self.input_size} of this model.")

        try:
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            result = self.model.run([output_name], {input_name: input_tensor.numpy()})

            logits = get_logits(result[0], ModelsTypes.ONNX)
            pixels_distribution = get_pixel_category_distribution(logits)
            seg_map_image, seg_map = get_segmentation_map(logits, self.input_size)

            return logits, pixels_distribution, seg_map, seg_map_image
        except Exception as e:
            raise RuntimeError(f"Error during ONNX model inference: {e}")

    @staticmethod
    def get_default_onnx_converted_segmentation_model(onnx_model_path: Path = PHASE_1_ONNX_MODEL_PATH) -> ort.InferenceSession:
        onnx_model = ort.InferenceSession(onnx_model_path)
        return onnx_model


class SegmentationModelAI:
    """
    SegmentationModelAI: High-level interface to perform inference with segmentation models.

    This class unifies the process of loading and running inference across different model platforms (PyTorch, ONNX).
    To support additional model types, refer to the BaseModel class docstring for implementation
    details. No modifications to SegmentationModelAI are necessary.
    """

    def __init__(self, model: Union[str, object], model_type: str, input_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> None:
        """
        Initialize the SegmentationModelAI instance with a model and its type.

        Parameters:
        -----------
        model : str or object
            Path to the model file or model object itself.
        model_type : str
            Type of the model ('torch', 'onnx'). More types can be added
            by implementing the required functionality.
        input_size : tuple, optional
            Input size of the image (default is (512, 512)).

        Raises:
        -------
        ValueError if the model type is not registered.
        """
        self.model = ModelRegistry.create_model(model, model_type, input_size)
        self.input_size = input_size

    def __call__(self, image: Union[str, bytes, Image.Image, torch.Tensor], resize_to: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Image.Image]:
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
        ValueError if the image input type is invalid.
        RuntimeError if an error occurs during image preprocessing or model inference.
        """

        image = ImageProcessor.load_image(image)
        ImageProcessor.validate_image(image)
        input_tensor = ImageProcessor.preprocess_image(image, resize_to)

        try:
            return self.model.infer(input_tensor)
        except ValueError as e:
            raise ValueError(f"Error during SegmentationModelAI inference: {e}. input_tensor value is invalid.")
        except Exception as e:
            raise RuntimeError(f"Error during inference: {e}")


def main():
    # example of simple use of SegmentationModelAI with different images / models types:

    torch_model = TorchModel.get_default_torch_segmentation_model()

    # Using Torch model
    model = SegmentationModelAI(torch_model, model_type='torch', input_size=(512, 512))
    logits, pixels_distribution, seg_map, seg_map_image = model('images/dog1.jpg')  # Local image
    results = model(
        'https://pickture.co.il/wp-content/uploads/2023/04/%D7%AA%D7%9E%D7%95%D7%A0%D7%94-%D7%A9%D7%9C-%D7%9B%D7%9C'
        '%D7%91-15-768x768.jpg')  # Image URL
    # results = model(ImageProcessor.image_to_bytes("images/dog1.jpg"))  # Image bytes

    # print / save / display the results (:
    seg_map_image.save("output_images/SegmentationModelAI_main_example.jpg")
    plt.imshow(seg_map_image)
    plt.show()
    print(logits)


    onnx_model = ONNXModel.get_default_onnx_converted_segmentation_model()

    # Using ONNX model
    model = SegmentationModelAI(onnx_model, model_type='onnx', input_size=(512, 512))
    logits, pixels_distribution, seg_map, seg_map_image = model('images/dog1.jpg')  # Local image
    # results = model(
    #     'https://pickture.co.il/wp-content/uploads/2023/04/%D7%AA%D7%9E%D7%95%D7%A0%D7%94-%D7%A9%D7%9C-%D7%9B%D7%9C'
    #     '%D7%91-15-768x768.jpg')  # Image URL
    # results = model(ImageProcessor.image_to_bytes("images/dog1.jpg"))  # Image bytes

    # print / save / display the results (:

    # seg_map_image.save("output_images/SegmentationModelAI_main_example.jpg")
    # plt.imshow(seg_map_image)
    # plt.show()
    # print(logits)


if __name__ == '__main__':
    main()
