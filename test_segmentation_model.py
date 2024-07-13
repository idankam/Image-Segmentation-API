import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import torch
import requests
from io import BytesIO
from SegmentationModelAI import ImageProcessor, SegmentationModelAI, TorchModel, ONNXModel
import torch
import onnxruntime as ort
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
from abc import ABC, abstractmethod


class TestImageProcessor(unittest.TestCase):

    def test_load_image_from_path_or_url(self):
        # Mock the requests.get call to return a fake image
        with patch('requests.get') as mock_get:
            mock_get.return_value.content = BytesIO(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR').getvalue()
            mock_get.return_value.status_code = 200
            image = ImageProcessor.load_image_from_path_or_url('https://example.com/fake_image.png')
            self.assertIsInstance(image, Image.Image)

    def test_load_image_from_bytes(self):
        # Mock image bytes
        image_bytes = BytesIO(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR').getvalue()
        image = ImageProcessor.load_image_from_bytes(image_bytes)
        self.assertIsInstance(image, Image.Image)

    def test_preprocess_image(self):
        # Mock an image file path
        with patch('PIL.Image.open') as mock_open:
            mock_open.return_value = Image.new('RGB', (512, 512))
            tensor = ImageProcessor.preprocess_image('path/to/image.jpg')
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tensor.shape, (1, 3, 512, 512))


class TestSegmentationModelAI(unittest.TestCase):
    def test_torch_inference(self):
        model = TorchModel.get_default_torch_segmentation_model()
        seg_model_ai = SegmentationModelAI(model, 'torch')

        # Mock an image
        image = Image.new('RGB', (512, 512))
        with patch.object(ImageProcessor, 'load_image', return_value=image):
            with patch.object(ImageProcessor, 'preprocess_image', return_value=torch.randn(1, 3, 512, 512)):
                result = seg_model_ai(image)
                self.assertIsInstance(result, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
