import os

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

IMAGE_SIZE = 520
SRC_DIR_PATH = 'images'
DST_DIR_PATH = 'output_images'
SEG_PREFIX = 'COCO_seg_'


# Load and preprocess the image
# Note: Normalize using values derived from the statistics of ImageNet Dataset.
# Those values have been widely adopted across various deep learning frameworks and pretrained models
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize image to 520x520 (DeepLabv3 input size)
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Post-process the output to create a segmentation map
def postprocess_output(output, original_image_size):
    output = output['out'][0].detach().cpu().numpy()
    output = output.argmax(0)
    output = Image.fromarray(output.astype(np.uint8)).resize(original_image_size)
    return output


#  run and save each image after segmentation (original and segmented together for comparison).
#  although tensor and onnx models can process multiple images simultaneously using batched inputs,
#  this exercise is for one image at a time, so i didn't use this method.
def segment_images_in_directory(directory_source_path, directory_destination_path, model):
    for filename in os.listdir(directory_source_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Load and preprocess the image
            image_path = os.path.join(directory_source_path, filename)
            input_image = preprocess_image(image_path)
            original_image = Image.open(image_path)
            original_size = original_image.size

            # Step 3: Perform inference
            with torch.no_grad():
                output = model(input_image)

            # Step 4: Post-process the output to create a segmentation map
            segmentation_map = postprocess_output(output, original_size)

            # Step 5: Visualize the result
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title('Original Image')
            plt.imshow(original_image)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title('Segmentation Map')
            plt.imshow(segmentation_map)
            plt.axis('off')

            seg_image_path = os.path.join(directory_destination_path, f'{SEG_PREFIX}{filename}')
            plt.savefig(seg_image_path, bbox_inches='tight')


# Example usage
def main():
    # Step 1: Load DeepLabv3 model
    weights = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT  # Get the most up-to-date weights
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
    model.eval()  # Set model to evaluation mode

    segment_images_in_directory(SRC_DIR_PATH, DST_DIR_PATH, model)


if __name__ == '__main__':
    main()
