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
    @staticmethod
    def preprocess_image(image, size=(512, 512)):
        """
        Preprocess the input image by resizing, normalizing, and converting to tensor.
        """
        image = ImageProcessor.load_image(image)
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
        if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path_or_url).convert('RGB')
        return image

    @staticmethod
    def load_image_from_bytes(image_bytes):
        """
        Load the image from bytes.
        """
        return Image.open(BytesIO(image_bytes)).convert('RGB')

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


class BaseModel(ABC):
    def __init__(self, model, input_size=(512, 512)):
        self.model = model
        self.input_size = input_size

    @abstractmethod
    def infer(self, input_tensor):
        pass


class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(model_class):
            cls._registry[name] = model_class
            return model_class

        return decorator

    @classmethod
    def create_model(cls, model, model_type, input_size=(512, 512)):
        if model_type not in cls._registry:
            raise ValueError(f"Model type '{model_type}' is not registered.")

        # initialize accordingly the type of the model
        return cls._registry[model_type](model, input_size)


@ModelRegistry.register('torch')
class TorchModel(BaseModel):
    def __init__(self, model, input_size=(512, 512)):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Provided model is not a valid PyTorch model.")
        super().__init__(model, input_size)

    def infer(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
        return output


@ModelRegistry.register('onnx')
class ONNXModel(BaseModel):
    def __init__(self, model, input_size=(512, 512)):
        if isinstance(model, str):
            try:
                model = ort.InferenceSession(model)
            except Exception as e:
                raise ValueError(f"Failed to load ONNX model: {e}")
        elif not isinstance(model, ort.InferenceSession):
            raise TypeError("Provided model is not a valid ONNX InferenceSession or path to an ONNX model file.")
        super().__init__(model, input_size)

    def infer(self, input_tensor):
        ort_inputs = {self.model.get_inputs()[0].name: input_tensor.numpy()}
        ort_outputs = self.model.run(None, ort_inputs)
        return ort_outputs[0]


@ModelRegistry.register('tensorflow')
class TensorFlowModel(BaseModel):
    def __init__(self, model, input_size=(512, 512)):
        if isinstance(model, str):
            try:
                model = tf.saved_model.load(model)
            except Exception as e:
                raise ValueError(f"Failed to load TensorFlow model: {e}")
        elif not isinstance(model, tf.Module):
            raise TypeError("Provided model is not a valid TensorFlow model or path to a TensorFlow model directory.")
        super().__init__(model, input_size)

    def infer(self, input_tensor):
        # Convert PyTorch tensor to NumPy array for TensorFlow
        input_array = input_tensor.numpy()
        input_array = tf.convert_to_tensor(input_array)

        # Assuming the TensorFlow model expects a dictionary input
        outputs = self.model(input_array, training=False)

        # Assuming single output
        return outputs[0].numpy()


class SegmentationModelAI:
    """
    NOTE: The SegmentationModelAI class uses ModelRegistry.create_model to create the model, and it does not handle any
    specific model type logic. This ensures that any new model class can be added without modifying SegmentationModelAI.
    """

    def __init__(self, model, model_type, input_size=(512, 512)):
        self.model = ModelRegistry.create_model(model, model_type, input_size)
        self.input_size = input_size

    def __call__(self, image):
        input_tensor = ImageProcessor.preprocess_image(image, self.input_size)
        return self.model.infer(input_tensor)


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
