import os
import time

import torch
import torchvision.transforms as transforms
from torchvision import models
import onnx
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

IMAGE_SIZE = 512
ONNX_MODEL_PATH = 'onnx_models/deeplabv3_mobilenet_v3_large.onnx'
IMAGES_DIR = 'images'
OUTPUT_IMAGES_DIR = 'output_images'
IMAGE_NAME = 'Roads_cars.jpg'


class ModelsTypes(Enum):
    TORCH = 1
    ONNX = 2


# Load and preprocess the image from local storage
# Note: Normalize using values derived from the statistics of ImageNet Dataset.
# Those values have been widely adopted across various deep learning frameworks and pretrained models
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize image to 520x520 (DeepLabv3 input size)
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def get_pixel_category_distribution(output, model_type):
    if model_type is ModelsTypes.TORCH:
        return output['out'][0].detach().cpu().numpy()
    elif model_type is ModelsTypes.ONNX:
        return output[0]
    return None


# Post-process the output to create a segmentation map
def get_segmentation_map(pixel_category_distribution, original_image_size):
    seg_map = pixel_category_distribution.argmax(0)
    seg_map_original_size = Image.fromarray(seg_map.astype(np.uint8)).resize(original_image_size)
    return seg_map_original_size


def create_dummy_image_input(image_size=IMAGE_SIZE):
    dummy_input = torch.randn(1, 3, image_size, image_size)
    return dummy_input


# Convert the model to onnx
def convert_to_onnx(model, input_tensor, onnx_path):
    torch.onnx.export(model, input_tensor, onnx_path, export_params=True,
                      opset_version=11, do_constant_folding=True,
                      input_names=['input'], output_names=['output'])
    print(f"Model successfully converted to {onnx_path}")
    return onnx_path


def check_onnx_model(onnx_path):
    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model loaded and verified according to ONNX specification.")


# Run inference using onnxruntime
# assumption: check runtime for onnx running only (for example, excluding loading the model).
def run_onnx_inference(onnx_path, input_tensor):
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    start_time = time.time()
    result = session.run([output_name], {input_name: input_tensor.numpy()})
    runtime = time.time() - start_time

    return result[0], runtime


# Get the most up-to-date DeepLabV3_MobileNet_V3_Large_Weights model
def get_torch_segmentation_model():
    weights = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT  # Get the most up-to-date weights
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
    model.eval()  # Set model to evaluation mode
    return model


def get_tensor_input_from_local_image(directory_image_path, image_file_name):
    image_path = os.path.join(directory_image_path, image_file_name)
    tensor_image = preprocess_image(image_path)
    original_image = Image.open(image_path)
    original_image_size = original_image.size
    return tensor_image, original_image, original_image_size


def run_torch_model(model, input_image):
    with torch.no_grad():
        start_time = time.time()
        output = model(input_image)
        runtime = time.time() - start_time

    return output, runtime


def L2(torch_output, onnx_output):
    l2_diff = np.linalg.norm(torch_output - onnx_output)
    return l2_diff


# Function to visualize the results
def visualize_results(original_image, torch_segmentation_map, onnx_segmentation_map, torch_runtime, onnx_runtime,
                      l2_result, directory_destination_path, filename):
    # Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    # Original image in the first row, centered
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].text(0.5, -0.1, f'Runtime Difference: {abs(torch_runtime - onnx_runtime):.4f}s', ha='center',
                 transform=axes[0].transAxes)
    axes[0].text(0.5, -0.2, f'L2: {l2_result:.4f}', ha='center', transform=axes[0].transAxes)
    axes[0].axis('off')

    # Torch model segmentation map
    axes[1].imshow(torch_segmentation_map)
    axes[1].set_title('Torch Model')
    axes[1].text(0.5, -0.1, f'Runtime: {torch_runtime:.4f}s', ha='center', transform=axes[1].transAxes)
    axes[1].axis('off')

    # ONNX model segmentation map
    axes[2].imshow(onnx_segmentation_map)
    axes[2].set_title('ONNX Model')
    axes[2].text(0.5, -0.1, f'Runtime: {onnx_runtime:.4f}s', ha='center', transform=axes[2].transAxes)
    axes[2].axis('off')

    # Adjust layout to remove unnecessary blanks
    plt.tight_layout()

    # Save and show the figure
    seg_image_path = os.path.join(directory_destination_path, f'{filename}')
    plt.savefig(seg_image_path, bbox_inches='tight')
    plt.show()


def print_result(torch_model_runtime, onnx_model_runtime, l2_result):
    print(f"torch Runtime: {torch_model_runtime:.4f}s")
    print(f"onnx Runtime: {onnx_model_runtime:.4f}s")
    print(f"Runtime diff (absolute): {abs(torch_model_runtime - onnx_model_runtime):.4f}s")
    print(f"L2 difference between models output: {l2_result}")


def main():
    # crate torch model:
    torch_model = get_torch_segmentation_model()

    # convert to onnx model:
    dummy_input = create_dummy_image_input(image_size=IMAGE_SIZE)
    onnx_path = convert_to_onnx(model=torch_model, input_tensor=dummy_input, onnx_path=ONNX_MODEL_PATH)
    check_onnx_model(onnx_path=onnx_path)

    # create input image:
    tensor_image, original_image, original_image_size = get_tensor_input_from_local_image(IMAGES_DIR, IMAGE_NAME)

    # run tensor model:
    torch_model_output, torch_model_runtime = run_torch_model(torch_model, tensor_image)

    # run onnx model:
    onnx_model_output, onnx_model_runtime = run_onnx_inference(onnx_path=onnx_path, input_tensor=tensor_image)

    # get pixels category distribution
    torch_output_pixels_distribution = get_pixel_category_distribution(torch_model_output, ModelsTypes.TORCH)
    onnx_output_pixels_distribution = get_pixel_category_distribution(onnx_model_output, ModelsTypes.ONNX)

    # get segmentation map (by choosing the highest category probability for each pixel)
    onnx_segmentation_map = get_segmentation_map(onnx_output_pixels_distribution, original_image_size)
    torch_segmentation_map = get_segmentation_map(torch_output_pixels_distribution, original_image_size)

    # get L2 between model output:
    l2_result = L2(torch_output_pixels_distribution, onnx_output_pixels_distribution)

    # print and visualize results:
    print_result(torch_model_runtime, onnx_model_runtime, l2_result)
    visualize_results(original_image, torch_segmentation_map, onnx_segmentation_map, torch_model_runtime,
                      onnx_model_runtime, l2_result, OUTPUT_IMAGES_DIR, "image_comparison")


if __name__ == '__main__':
    main()
