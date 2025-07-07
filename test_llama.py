import requests

# Define the URL of the Flask server
url = "http://localhost:5001/generate_description"

# Define the path to the image you want to send
image_path = "./Image/Cats.jpg"  # Replace this with your image path

# Define the prompt text
prompt = "What is this?"

# Open the image file in binary mode
with open(image_path, 'rb') as image_file:
    # Prepare the files and data to send in the POST request
    files = {'image': image_file}
    data = {'prompt': prompt}

    # Send the POST request to the server
    response = requests.post(url, files=files, data=data)

    # Check if the request was successful
    if response.status_code == 200:
        # Print the description returned by the server
        print("Description:", response.json()['description'])
    else:
        # Print the error if the request was not successful
        print("Failed to get a description. Server returned:", response.text)
