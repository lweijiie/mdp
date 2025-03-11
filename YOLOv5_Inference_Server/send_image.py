import requests
import os

# Set the URL of the Flask server
url = "http://localhost:5000/image"

# Path to the images folder
images_folder = "C:/Users/DELL G15/Desktop/SCHOOL/MDP/mdp/YOLOv5_Inference_Server/uploads"

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Check current working directory
print("Current Working Directory:", os.getcwd())
print(f"Found {len(image_files)} images in folder: {images_folder}")

# Loop through each image and send it to the server
for image_name in image_files:
    image_path = os.path.join(images_folder, image_name)
    print(f"\nSending image: {image_path}")

    try:
        with open(image_path, "rb") as image_file:
            files = {"file": (image_name, image_file, "image/jpg")}
            response = requests.post(url, files=files)

        # Print raw response
        print("Status Code:", response.status_code)
        print("Raw Response:", response.text)

        # Try parsing the response as JSON
        try:
            json_data = response.json()
            print(f"Response JSON: {json_data}")
        except requests.exceptions.JSONDecodeError:
            print("Response is not in JSON format!")

    except Exception as e:
        print(f"Error processing {image_name}: {e}")

print("\nFinished sending all images.")
