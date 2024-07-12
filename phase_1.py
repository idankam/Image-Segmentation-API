import time
from enum import Enum
from pathlib import Path
from typing import Union, Tuple

import numpy as np  # ONNX Runtime Python packages now have numpy dependency >=1.21.6, <2.0.
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as torch_func
import onnx
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt


# Constants
IMAGE_SIZE = 512
PHASE_1_ONNX_MODEL_PATH = Path('onnx_models/deeplabv3_mobilenet_v3_large.onnx')
IMAGES_DIR = Path('images')
OUTPUT_IMAGES_DIR = Path('output_images')
IMAGE_NAME = 'Dog3.jpeg'


class ModelsTypes(Enum):
    TORCH = 1
    ONNX = 2


def preprocess_image(image_path: Path) -> torch.Tensor:
    """
    Load and preprocess the image from local storage.
    Normalize using values derived from the statistics of ImageNet Dataset.
    """
    image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize image to 512x512
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def get_logits(output, model_type: ModelsTypes) -> np.ndarray:
    """
    Extract pixel category distribution from model output.
    """
    if model_type == ModelsTypes.TORCH:
        return output['out'][0].detach().cpu().numpy()
    elif model_type == ModelsTypes.ONNX:
        return output[0]
    raise ValueError(f"Invalid model type '{model_type}'. Expected ModelsTypes.TORCH or ModelsTypes.ONNX.")


def get_pixel_category_distribution(logits) -> np.ndarray:
    """
    Extract pixel category distribution probabilities from logits.
    """
    softmax_output = torch_func.softmax(torch.from_numpy(logits), dim=0)
    return softmax_output.numpy()


def get_segmentation_map(logits_output: np.ndarray, original_image_size: tuple[int, int]) -> Image:
    """
    Generate segmentation map from logits output.
    """
    seg_map = logits_output.argmax(0)
    seg_map = seg_map.astype(np.uint8)
    seg_map_image_original_size = Image.fromarray(seg_map.astype(np.uint8)).resize(original_image_size)
    return seg_map_image_original_size, seg_map


def create_dummy_image_input(image_size: Union[int, Tuple[int, int]] = IMAGE_SIZE) -> torch.Tensor:
    """
    Create a dummy input tensor for model conversion and testing.
    """
    if isinstance(image_size, int):
        # If image_size is an integer, assume square image with equal width and height
        return torch.randn(1, 3, image_size, image_size)
    elif isinstance(image_size, tuple) and len(image_size) == 2:
        # If image_size is a tuple with (width, height)
        width, height = image_size
        return torch.randn(1, 3, height, width)  # assuming height comes first in PyTorch convention
    else:
        raise ValueError("image_size should be an integer or a tuple of two integers (width, height)")


def convert_to_onnx(model: torch.nn.Module, input_tensor: torch.Tensor, onnx_path: Path) -> Path:
    """
    Convert a PyTorch model to ONNX format and save it.
    """
    torch.onnx.export(model, input_tensor, onnx_path, export_params=True,
                      opset_version=11, do_constant_folding=True,
                      input_names=['input'], output_names=['output'])
    print(f"Model successfully converted to {onnx_path}")
    return onnx_path


def check_onnx_model(onnx_path: Path) -> None:
    """
    Check the validity of the ONNX model.
    """
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model loaded and verified according to ONNX specification.\n")


def run_onnx_inference(onnx_path: Path, input_tensor: torch.Tensor) -> tuple:
    """
    Run inference using ONNX model and measure runtime.
    """
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    start_time = time.time()
    result = session.run([output_name], {input_name: input_tensor.numpy()})
    runtime = time.time() - start_time

    return result[0], runtime


def get_torch_segmentation_model() -> torch.nn.Module:
    """
    Load the DeepLabV3 MobileNetV3 large model from torchvision.
    """
    weights = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
    model.eval()  # Set model to evaluation mode
    return model


def get_tensor_input_from_local_image(directory_image_path: Path, image_file_name: str) -> tuple:
    """
    Load and preprocess input image from local storage.
    """
    image_path = directory_image_path / image_file_name
    tensor_image = preprocess_image(image_path)
    original_image = Image.open(image_path)
    return tensor_image, original_image, original_image.size


def run_torch_model(model: torch.nn.Module, input_image: torch.Tensor) -> tuple:
    """
    Run inference using PyTorch model and measure runtime.
    """
    with torch.no_grad():
        start_time = time.time()
        output = model(input_image)
        runtime = time.time() - start_time

    return output, runtime


def L2(torch_output: np.ndarray, onnx_output: np.ndarray) -> float:
    """
    Calculate the L2 norm between two arrays.
    """
    return np.linalg.norm(torch_output - onnx_output)


def visualize_results(original_image: Image, torch_segmentation_map: Image, onnx_segmentation_map: Image,
                      torch_runtime: float, onnx_runtime: float, logits_l2_result: float, pixels_distribution_l2_result: float, seg_map_l2_result: float,
                      directory_destination_path: Path, filename: str) -> None:
    """
    Visualize and compare the results of Torch and ONNX models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].text(0.5, -0.1, f'Runtime Difference: {abs(torch_runtime - onnx_runtime):.5f}s', ha='center',
                 transform=axes[0].transAxes)
    axes[0].text(0.5, -0.15, f'L2 - logits: {logits_l2_result:.5f}', ha='center', transform=axes[0].transAxes)
    axes[0].text(0.5, -0.2, f'L2 - pixels distribution: {pixels_distribution_l2_result:.5f}', ha='center', transform=axes[0].transAxes)
    axes[0].text(0.5, -0.25, f'L2 - segmentation map: {seg_map_l2_result:.5f}', ha='center', transform=axes[0].transAxes)
    axes[0].axis('off')

    axes[1].imshow(torch_segmentation_map)
    axes[1].set_title('Torch Model')
    axes[1].text(0.5, -0.1, f'Runtime: {torch_runtime:.5f}s', ha='center', transform=axes[1].transAxes)
    axes[1].axis('off')

    axes[2].imshow(onnx_segmentation_map)
    axes[2].set_title('ONNX Model')
    axes[2].text(0.5, -0.1, f'Runtime: {onnx_runtime:.5f}s', ha='center', transform=axes[2].transAxes)
    axes[2].axis('off')

    seg_image_path = directory_destination_path / filename
    plt.savefig(seg_image_path, bbox_inches='tight')
    plt.show()


def print_result(torch_model_runtime: float, onnx_model_runtime: float, logits_l2_result: float,
                 pixels_distribution_l2_result: float, seg_map_l2_result: float) -> None:
    """
    Print the runtime and L2 difference between Torch and ONNX models.
    """
    print()
    print(f"torch Runtime: {torch_model_runtime:.4f}s")
    print(f"onnx Runtime: {onnx_model_runtime:.4f}s")
    print(f"Runtime diff (absolute): {abs(torch_model_runtime - onnx_model_runtime):.4f}s")
    print(f"L2 difference between models logits output: {logits_l2_result}")
    print(f"L2 difference between models pixels distribution output: {pixels_distribution_l2_result}")
    print(f"L2 difference between models segmentation map output: {seg_map_l2_result}")


def check_pixel_probs_sum_to_one(probabilities: np.ndarray, model_type: ModelsTypes) -> bool:
    # probabilities are assumed to be a numpy array of shape (C, H, W) where C is the number of classes

    # Sum probabilities across the class dimension (axis 0)
    sum_probabilities = np.sum(probabilities, axis=0)

    # Check if sum of probabilities at each pixel equals 1 (within a small tolerance)
    tolerance = 1e-4  # Adjust tolerance as needed
    all_sum_to_one = np.allclose(sum_probabilities, np.ones_like(sum_probabilities), atol=tolerance)

    print(f'{model_type.name} output -> each pixel distribution probabilities sum up to 1: {all_sum_to_one}')
    return all_sum_to_one


def main():
    # Load Torch model
    torch_model = get_torch_segmentation_model()

    # Convert to ONNX and check model
    dummy_input = create_dummy_image_input()
    onnx_path = convert_to_onnx(model=torch_model, input_tensor=dummy_input, onnx_path=PHASE_1_ONNX_MODEL_PATH)
    check_onnx_model(onnx_path)

    # Load input image
    tensor_image, original_image, original_image_size = get_tensor_input_from_local_image(IMAGES_DIR, IMAGE_NAME)

    # Run inference on Torch model
    torch_model_output, torch_model_runtime = run_torch_model(torch_model, tensor_image)

    # Run inference on ONNX model
    onnx_model_output, onnx_model_runtime = run_onnx_inference(onnx_path, tensor_image)

    # Get logits
    torch_logits_output = get_logits(torch_model_output, ModelsTypes.TORCH)
    onnx_logits_output = get_logits(onnx_model_output, ModelsTypes.ONNX)

    # Get pixel category distribution probabilities
    torch_output_pixels_distribution = get_pixel_category_distribution(torch_logits_output)
    onnx_output_pixels_distribution = get_pixel_category_distribution(onnx_logits_output)

    # validate pixel category distribution probabilities
    check_pixel_probs_sum_to_one(torch_output_pixels_distribution, ModelsTypes.TORCH)
    check_pixel_probs_sum_to_one(onnx_output_pixels_distribution, ModelsTypes.ONNX)

    # Get segmentation maps
    torch_seg_map_image, torch_seg_map = get_segmentation_map(torch_logits_output, original_image_size)
    onnx_seg_map_image, onnx_seg_map = get_segmentation_map(onnx_logits_output, original_image_size)

    # Calculate L2 differences
    logits_l2_result = L2(torch_logits_output, onnx_logits_output)
    pixels_distribution_l2_result = L2(torch_output_pixels_distribution, onnx_output_pixels_distribution)
    seg_map_l2_result = L2(torch_seg_map, onnx_seg_map)

    # Print and visualize results
    print_result(torch_model_runtime, onnx_model_runtime, logits_l2_result,
                 pixels_distribution_l2_result, seg_map_l2_result)
    visualize_results(original_image, torch_seg_map_image, onnx_seg_map_image, torch_model_runtime,
                      onnx_model_runtime, logits_l2_result, pixels_distribution_l2_result, seg_map_l2_result,
                      OUTPUT_IMAGES_DIR, "image_comparison.png")


if __name__ == '__main__':
    main()
