import os
import requests

# Define the URL of your inference server
SERVER_URL = "http://localhost:5000/image"

# Define the path to the directory containing the images you want to test
IMAGE_DIR = "C:/Users/DELL G15/Desktop/SCHOOL/MDP/mdp/uploads"

def send_image_to_server(image_path):
    """
    Send an image to the inference server and get the prediction result.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: Prediction result from the server.
    """
    try:
        # Open the image file in binary mode
        with open(image_path, 'rb') as file:
            # Prepare the files dictionary for the POST request
            files = {'file': (os.path.basename(image_path), file, 'image/jpeg')}
            
            # Send the POST request to the server
            response = requests.post(SERVER_URL, files=files)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Return the JSON response
                return response.json()
            else:
                print(f"Error: Received status code {response.status_code}")
                return None
    except Exception as e:
        print(f"Error sending image {image_path}: {e}")
        return None

def test_images_in_directory(directory):
    """
    Test all images in a directory by sending them to the inference server.

    Args:
        directory (str): Path to the directory containing images.
    """
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        # Check if the file is an image (you can add more extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Sending image: {filename}")
            
            # Send the image to the server and get the prediction
            result = send_image_to_server(file_path)
            
            # Print the prediction result
            if result:
                print(f"Prediction for {filename}: {result}")
            else:
                print(f"Failed to get prediction for {filename}")
            print("-" * 40)

if __name__ == "__main__":
    # Test all images in the specified directory
    test_images_in_directory(IMAGE_DIR)