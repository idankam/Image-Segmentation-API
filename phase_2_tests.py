import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import onnxruntime as ort
from phase_2 import *
from phase_1 import check_pixel_probs_sum_to_one, ModelsTypes, create_dummy_image_input, convert_to_onnx, \
    check_onnx_model

UNITTEST_ONNX_MODEL_PATH = Path('onnx_models/unittest_tmp_model.onnx')


def model_infer_checks(self, logits, pixels_distribution, seg_map, seg_map_image, model_type, size):
    self.assertIsNotNone(logits)
    self.assertIsNotNone(pixels_distribution)
    self.assertIsNotNone(seg_map)
    self.assertIsNotNone(seg_map_image)
    self.assertIsInstance(logits, np.ndarray)
    self.assertIsInstance(pixels_distribution, np.ndarray)
    self.assertIsInstance(seg_map, np.ndarray)
    self.assertIsInstance(seg_map_image, Image.Image)
    self.assertTupleEqual(logits.shape, (21, size[0], size[1]))
    self.assertTupleEqual(pixels_distribution.shape, (21, size[0], size[1]))
    self.assertTupleEqual(seg_map.shape, size)
    self.assertTupleEqual(seg_map_image.size, size)
    self.assertTrue(check_pixel_probs_sum_to_one(pixels_distribution, model_type))
    self.assertTrue(len(np.unique(seg_map)) > 1)


class TestImageProcessor(unittest.TestCase):
    def test_load_image_from_path(self):
        image_path = 'images/people.jpg'
        image = ImageProcessor.load_image(image_path)
        self.assertIsNotNone(image)
        self.assertIsInstance(image, Image.Image)

    def test_load_image_from_url(self):
        image_url = 'https://pickture.co.il/wp-content/uploads/2023/04/%D7%AA%D7%9E%D7%95%D7%A0%D7%94-%D7%A9%D7%9C-%D7%9B%D7%9C%D7%91-15-768x768.jpg'
        image = ImageProcessor.load_image(image_url)
        self.assertIsNotNone(image)
        self.assertIsInstance(image, Image.Image)

    def test_load_image_from_bytes(self):
        with open('images/people.jpg', 'rb') as f:
            image_bytes = f.read()
        image = ImageProcessor.load_image(image_bytes)
        self.assertIsNotNone(image)
        self.assertIsInstance(image, Image.Image)

    def test_preprocess_image(self):
        image = ImageProcessor.load_image('images/people.jpg')
        preprocessed_image = ImageProcessor.preprocess_image(image, size=(512, 512))
        self.assertIsNotNone(preprocessed_image)
        self.assertIsInstance(preprocessed_image, torch.Tensor)
        self.assertTupleEqual(preprocessed_image.shape, (1, 3, 512, 512))

    def test_load_tensor_image_3d(self):
        dummy_image = torch.rand((3, 512, 512))  # Example tensor image (channels, height, width)
        loaded_image = ImageProcessor.load_image(dummy_image)
        self.assertTrue(torch.equal(loaded_image, dummy_image.unsqueeze(0)))

    def test_load_tensor_image_already_batched(self):
        dummy_image = torch.rand((1, 3, 512, 512))  # Example tensor image (1, channels, height, width)
        loaded_image = ImageProcessor.load_image(dummy_image)
        self.assertTrue(torch.equal(loaded_image, dummy_image))

    def test_preprocess_tensor_image(self):
        dummy_image = torch.rand((3, 512, 512))  # Example tensor image (channels, height, width)
        processed_tensor = ImageProcessor.preprocess_image(dummy_image)
        expected_shape = (1, 3, DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1])
        self.assertEqual(processed_tensor.shape, expected_shape)

    def test_image_processor_invalid_input_type(self):
        with self.assertRaises(TypeError):
            ImageProcessor.load_image(1234)  # int is not supported

    def test_invalid_image_path(self):
        with self.assertRaises(RuntimeError):
            ImageProcessor.load_image('invalid_image.jpg')

    def test_invalid_image_url(self):
        with self.assertRaises(RuntimeError):
            ImageProcessor.load_image('http://broken_url.co.il')
            ImageProcessor.load_image('https://broken_url.co.il')

    def test_preprocess_image_different_sizes(self):
        image = ImageProcessor.load_image('images/people.jpg')
        sizes = [(1, 1), (32, 32), (256, 256), (512, 512), (1024, 1024), (8096, 8096)]
        for size in sizes:
            preprocessed_image = ImageProcessor.preprocess_image(image, size=size)
            self.assertIsNotNone(preprocessed_image)
            self.assertIsInstance(preprocessed_image, torch.Tensor)
            self.assertTupleEqual(preprocessed_image.shape, (1, 3, size[0], size[1]))

    def test_preprocess_image_invalid_sizes(self):
        image = ImageProcessor.load_image('images/people.jpg')
        sizes = [(0, 0), (0, 1), (1, 0), (-1, 512), (512, -1), (-1, -1)]
        for size in sizes:
            with self.assertRaises(ValueError):
                ImageProcessor.preprocess_image(image, size=size)


class TestTorchModel(unittest.TestCase):
    def setUp(self):
        self.torch_model = TorchModel(TorchModel.get_default_torch_segmentation_model(), input_size=(512, 512))

    def test_torch_model_infer(self):
        image = ImageProcessor.load_image('images/people.jpg')
        input_tensor = ImageProcessor.preprocess_image(image, size=(512, 512))
        logits, pixels_distribution, seg_map, seg_map_image = self.torch_model.infer(input_tensor)
        model_infer_checks(self, logits, pixels_distribution, seg_map, seg_map_image, ModelsTypes.TORCH, (512, 512))

    def test_get_torch_segmentation_model(self):
        model = TorchModel.get_default_torch_segmentation_model()
        self.assertIsNotNone(model)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertIsInstance(model, models.segmentation.deeplabv3.DeepLabV3)

    def test_invalid_model_type(self):
        with self.assertRaises(TypeError):
            TorchModel('invalid_model')

        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)  # CLASSIFICATION MODEL!
        model.eval()
        with self.assertRaises(TypeError):
            TorchModel(model)

    def test_torch_model_infer_different_sizes(self):
        image = ImageProcessor.load_image('images/people.jpg')
        sizes = [(256, 256), (512, 512), (1024, 1024)]
        for size in sizes:
            torch_model = TorchModel(TorchModel.get_default_torch_segmentation_model(), input_size=size)
            input_tensor = ImageProcessor.preprocess_image(image, size=size)
            logits, pixels_distribution, seg_map, seg_map_image = torch_model.infer(input_tensor)
            model_infer_checks(self, logits, pixels_distribution, seg_map, seg_map_image, ModelsTypes.TORCH, size)

    def test_infer_input_size_mismatch(self):
        torch_model = TorchModel(TorchModel.get_default_torch_segmentation_model(), input_size=(512, 512))

        # Create a sample input tensor with a different size
        input_tensor = torch.randn(1, 3, 256, 256)  # Incorrect size, should be (512, 512)

        # Check that infer method raises a ValueError
        with self.assertRaises(ValueError):
            torch_model.infer(input_tensor)


class TestONNXModel(unittest.TestCase):
    def setUp(self):
        self.onnx_model = ONNXModel(ONNXModel.get_default_onnx_converted_segmentation_model(), input_size=(512, 512))

    def test_onnx_model_infer(self):
        image = ImageProcessor.load_image('images/people.jpg')
        input_tensor = ImageProcessor.preprocess_image(image, size=(512, 512))
        logits, pixels_distribution, seg_map, seg_map_image = self.onnx_model.infer(input_tensor)
        model_infer_checks(self, logits, pixels_distribution, seg_map, seg_map_image, ModelsTypes.ONNX, (512, 512))

    def test_get_onnx_converted_segmentation_model(self):
        model = ONNXModel.get_default_onnx_converted_segmentation_model()
        self.assertIsNotNone(model)
        self.assertIsInstance(model, ort.InferenceSession)

    def test_invalid_model_type(self):
        with self.assertRaises(RuntimeError):
            ONNXModel('invalid_model.onnx')

        torch_model = TorchModel.get_default_torch_segmentation_model()
        with self.assertRaises(TypeError):
            ONNXModel(torch_model)

    def test_onnx_model_infer_different_sizes(self):
        image = ImageProcessor.load_image('images/people.jpg')
        sizes = [(256, 256), (512, 512), (1024, 1024)]
        for size in sizes:
            # Load Torch model
            torch_model = TorchModel.get_default_torch_segmentation_model()

            # Convert to ONNX and check model
            dummy_input = create_dummy_image_input(image_size=size)
            convert_to_onnx(model=torch_model, input_tensor=dummy_input, onnx_path=UNITTEST_ONNX_MODEL_PATH)
            check_onnx_model(UNITTEST_ONNX_MODEL_PATH)

            input_tensor = ImageProcessor.preprocess_image(image, size=size)
            onnx_model = ONNXModel(UNITTEST_ONNX_MODEL_PATH, input_size=size)

            logits, pixels_distribution, seg_map, seg_map_image = onnx_model.infer(input_tensor)
            model_infer_checks(self, logits, pixels_distribution, seg_map, seg_map_image, ModelsTypes.ONNX, size)

    def test_infer_input_size_mismatch(self):
        onnx_model = ONNXModel(ONNXModel.get_default_onnx_converted_segmentation_model(), input_size=(512, 512))

        # Create a sample input tensor with a different size
        input_tensor = torch.randn(1, 3, 256, 256)  # Incorrect size, should be (512, 512)

        # Check that infer method raises a ValueError
        with self.assertRaises(ValueError):
            onnx_model.infer(input_tensor)


class TestSegmentationModelAI(unittest.TestCase):
    def setUp(self):
        self.torch_model = TorchModel.get_default_torch_segmentation_model()
        self.onnx_model = ONNXModel.get_default_onnx_converted_segmentation_model()

    def test_segmentation_model_ai_torch(self):
        model = SegmentationModelAI(self.torch_model, model_type='torch', input_size=(512, 512))
        logits, pixels_distribution, seg_map, seg_map_image = model('images/people.jpg')
        model_infer_checks(self, logits, pixels_distribution, seg_map, seg_map_image, ModelsTypes.TORCH, (512, 512))

    def test_segmentation_model_ai_onnx(self):
        model = SegmentationModelAI(self.onnx_model, model_type='onnx', input_size=(512, 512))
        logits, pixels_distribution, seg_map, seg_map_image = model('images/people.jpg')
        model_infer_checks(self, logits, pixels_distribution, seg_map, seg_map_image, ModelsTypes.ONNX, (512, 512))

    def test_invalid_model_type(self):
        with self.assertRaises(ValueError):
            SegmentationModelAI('invalid_model', model_type='invalid_type', input_size=(512, 512))

    def test_invalid_image_path_torch(self):
        model = SegmentationModelAI(self.torch_model, model_type='torch', input_size=(512, 512))
        with self.assertRaises(RuntimeError):
            model('invalid_image.jpg')

    def test_invalid_image_path_onnx(self):
        model = SegmentationModelAI(self.onnx_model, model_type='onnx', input_size=(512, 512))
        with self.assertRaises(RuntimeError):
            model('invalid_image.jpg')

    def test_segmentation_model_ai_torch_different_sizes(self):
        sizes = [(256, 256), (512, 512), (1024, 1024)]
        for size in sizes:
            model = SegmentationModelAI(self.torch_model, model_type='torch', input_size=size)
            logits, pixels_distribution, seg_map, seg_map_image = model('images/people.jpg', size)
            model_infer_checks(self, logits, pixels_distribution, seg_map, seg_map_image, ModelsTypes.TORCH, size)

    def test_segmentation_model_ai_onnx_different_sizes(self):
        global UNITTEST_ONNX_MODEL_PATH
        sizes = [(256, 256), (512, 512), (1024, 1024)]
        for size in sizes:
            # Load Torch model
            torch_model = TorchModel.get_default_torch_segmentation_model()

            # Convert to ONNX and check model
            dummy_input = create_dummy_image_input(image_size=size)
            convert_to_onnx(model=torch_model, input_tensor=dummy_input, onnx_path=UNITTEST_ONNX_MODEL_PATH)
            check_onnx_model(UNITTEST_ONNX_MODEL_PATH)

            model = SegmentationModelAI(UNITTEST_ONNX_MODEL_PATH, model_type='onnx', input_size=size)
            logits, pixels_distribution, seg_map, seg_map_image = model('images/people.jpg', size)
            model_infer_checks(self, logits, pixels_distribution, seg_map, seg_map_image, ModelsTypes.ONNX, size)

    def test_segmentation_model_ai_torch_input_size_match(self):
        model = SegmentationModelAI(self.torch_model, model_type='torch', input_size=(512, 512))

        # Check that infer method raises a ValueError
        with self.assertRaises(ValueError):
            model('images/people.jpg', resize_to=(256, 256))  # Incorrect size, should be (512, 512)

    def test_segmentation_model_ai_onnx_input_size_match(self):
        model = SegmentationModelAI(self.onnx_model, model_type='onnx', input_size=(512, 512))

        # Check that infer method raises a ValueError
        with self.assertRaises(ValueError):
            model('images/people.jpg', resize_to=(256, 256))  # Incorrect size, should be (512, 512)

    def test_segmentation_model_ai_load_tensor(self):
        model = SegmentationModelAI(self.onnx_model, model_type='onnx', input_size=(512, 512))

        image = ImageProcessor.load_image('images/people.jpg')
        input_tensor = ImageProcessor.preprocess_image(image)

        model(input_tensor)


if __name__ == '__main__':
    unittest.main()
