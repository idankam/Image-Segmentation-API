import json
import numpy as np
import matplotlib.pyplot as plt
import requests


def infer_from_url(url, infer_model_type, requested_results_types, image_url):
    # Prepare the data
    data = {
        "infer_model_type": infer_model_type,
        "requested_results_types": requested_results_types,
        "url_image": image_url
    }

    # Send the POST request
    response = requests.post(url, data=data)

    # Handle the response
    if response.status_code == 200:
        print("Successful image inference from URL!")
        print("Status code:", response.status_code)
        with open('result_url.json', 'w') as fp:
            json.dump(response.json(), fp, indent=4)

        response_json = response.json()
        seg_map_image = np.array(response_json['requested_results']['segmentation_map'])
        plt.imshow(seg_map_image)
        plt.show()
    else:
        print("Failed image inference from URL")
        print("Status code:", response.status_code)
        print("Response:", response.text)


def infer_from_file(url, infer_model_type, requested_results_types, file_path):
    # Prepare the data and files
    data = {
        "infer_model_type": infer_model_type,
        "requested_results_types": requested_results_types,
    }
    files = {
        "file": (file_path.split('/')[-1], open(file_path, "rb"), "image/jpeg")
    }

    # Send the POST request
    response = requests.post(url, data=data, files=files)

    # Handle the response
    if response.status_code == 200:
        print("Successful image inference from file!")
        print("Status code:", response.status_code)
        with open('result_file.json', 'w') as fp:
            json.dump(response.json(), fp, indent=4)

        response_json = response.json()
        seg_map_image = np.array(response_json['requested_results']['segmentation_map'])
        plt.imshow(seg_map_image)
        plt.show()
    else:
        print("Failed image inference from file")
        print("Status code:", response.status_code)
        print("Response:", response.text)


# Example usage:
# URL endpoint for URL-based image inference
url_infer_url = "http://127.0.0.1:8000/infer/url/"

# URL-based image inference
infer_from_url(
    url=url_infer_url,
    infer_model_type="torch",
    requested_results_types=["segmentation_map"],
    image_url="https://pickture.co.il/wp-content/uploads/2023/04/%D7%AA%D7%9E%D7%95%D7%A0%D7%94-%D7%A9%D7%9C-%D7%9B%D7%9C%D7%91-15-768x768.jpg"
)

# URL endpoint for file-based image inference
url_infer_file = "http://127.0.0.1:8000/infer/file"

# File-based image inference
infer_from_file(
    url=url_infer_file,
    infer_model_type="torch",
    requested_results_types=["segmentation_map"],
    file_path="images/dog1.jpg"
)
