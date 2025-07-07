import requests
import base64
from PIL import Image
from io import BytesIO

# YOLO server URL
YOLO_SERVER_URL = "http://localhost:5001/process?type=image"

# Test function to send image and display the response
def test_yolo_with_bounding_boxes(image_path):
    # Open the image file
    with open(image_path, 'rb') as image_file:
        files = {'file': ('image.jpg', image_file, 'image/jpeg')}
        response = requests.post(YOLO_SERVER_URL, files=files)

    # Process the response
    if response.status_code == 200:
        data = response.json()
        detections = data['objects']
        print("Detections:", detections)

        # Decode the base64 image
        img_data = base64.b64decode(data['image'])
        img = Image.open(BytesIO(img_data))

        # Display the image with bounding boxes
        img.show()
    else:
        print("Failed to get response from YOLO server")

# Run the test
image_path = "./Image/Cats.jpg"  # Replace with your image path
test_yolo_with_bounding_boxes(image_path)
