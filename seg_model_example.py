import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

IMAGE_SIZE = 520


# Load and preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize image to 520x520 (DeepLabv3 input size)
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Post-process the output to create a segmentation map
def postprocess_output(output, original_image_size):
    output = output['out'][0].detach().cpu().numpy()
    output = output.argmax(0)
    output = Image.fromarray(output.astype(np.uint8)).resize(original_image_size)
    return output


# Example usage
def main():
    # Step 1: Load and preprocess the image
    image_path = 'images/dog1.jpg'
    input_image = preprocess_image(image_path)
    original_image = Image.open(image_path)
    original_size = original_image.size

    # Step 2: Load DeepLabv3 model
    weights = models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT  # Get the most up-to-date weights
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
    # model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    model.eval()  # Set model to evaluation mode

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

    # display in Jupyter Notebook / google colab:
    # plt.show()

    plt.savefig('dog1_seg.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
