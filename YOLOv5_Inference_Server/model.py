import os
import shutil
import time
import glob
import torch
from PIL import Image
from ultralytics import YOLO
import cv2
import random
import string
import numpy as np

def get_random_string(length):
    """
    Generate a random string of fixed length 

    Inputs
    ------
    length: int - length of the string to be generated

    Returns
    -------
    str - random string

    """
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    return result_str

def load_model():
    """
    Load the model from the local directory
    """
    model = YOLO("../Weights/bestv11.pt")
    print("Model Classes:", model.names)
    return model

def draw_own_bbox(img,x1,y1,x2,y2,label,color=(36,255,12),text_color=(0,0,0)):
    """
    Draw bounding box on the image with text label and save both the raw and annotated image in the 'own_results' folder

    Inputs
    ------
    img: numpy.ndarray - image on which the bounding box is to be drawn

    x1: int - x coordinate of the top left corner of the bounding box

    y1: int - y coordinate of the top left corner of the bounding box

    x2: int - x coordinate of the bottom right corner of the bounding box

    y2: int - y coordinate of the bottom right corner of the bounding box

    label: str - label to be written on the bounding box

    color: tuple - color of the bounding box

    text_color: tuple - color of the text label

    Returns
    -------
    None

    """
    name_to_id = {
        "NA": 'NA',
        "Bullseye": 10,
        "One": 11,
        "Two": 12,
        "Three": 13,
        "Four": 14,
        "Five": 15,
        "Six": 16,
        "Seven": 17,
        "Eight": 18,
        "Nine": 19,
        "A": 20,
        "B": 21,
        "C": 22,
        "D": 23,
        "E": 24,
        "F": 25,
        "G": 26,
        "H": 27,
        "S": 28,
        "T": 29,
        "U": 30,
        "V": 31,
        "W": 32,
        "X": 33,
        "Y": 34,
        "Z": 35,
        "Up": 36,
        "Down": 37,
        "Right": 38,
        "Left": 39,
        "Up Arrow": 36,
        "Down Arrow": 37,
        "Right Arrow": 38,
        "Left Arrow": 39,
        "Stop": 40
    }
    # Reformat the label to {label name}-{label id}
    label = label + "-" + str(name_to_id[label])
    # Convert the coordinates to int
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    # Create a random string to be used as the suffix for the image name, just in case the same name is accidentally used
    rand = str(int(time.time()))

    # Save the raw image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"own_results/raw_image_{label}_{rand}.jpg", img)

    # Draw the bounding box
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # For the text background, find space required by the text so that we can put a background with that amount of width.
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    # Print the text  
    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    img = cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
    # Save the annotated image
    cv2.imwrite(f"own_results/annotated_image_{label}_{rand}.jpg", img)

signal = "default_value"

def predict_image(image, model, signal):
    try:
        # Load the image
        img_path = os.path.join('images', image)
        print(f"Loading image from: {img_path}")
        img = Image.open(img_path)

        # Predict using the model
        print("Running model prediction...")
        results = model(img)
        print("Prediction completed.")

        # Access the first result in the list
        first_result = results[0]
        # Save the annotated prediction image with a unique filename
        annotated_frame = first_result.plot()

        # Extract the base name of the image (without extension)
        base_filename = os.path.splitext(image)[0]

        # Create a unique filename using the original image name
        prediction_filename = os.path.join('runs', f"{base_filename}_prediction.jpg")

        # Save the annotated image
        cv2.imwrite(prediction_filename, annotated_frame)
        print(f"Annotated image saved to '{prediction_filename}'.")

        # Extract detection boxes
        boxes = first_result.boxes
        if boxes is None or len(boxes) == 0:
            print("No objects detected.")
            return 'NA'

        # Extract bounding box coordinates, confidence, and class IDs
        boxes_data = boxes.xyxy.cpu().numpy()  # Bounding box coordinates: (x1, y1, x2, y2)
        confidences = boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = boxes.cls.cpu().numpy()  # Class IDs

        print(f"Boxes Data: {boxes_data}")
        print(f"Confidences: {confidences}")
        print(f"Class IDs: {class_ids}")

        # Map class IDs to class names
        class_names = first_result.names  # Dictionary mapping class ID to name
        print(f"Class Names Mapping: {class_names}")

        # Build predictions list
        predictions = []
        for i in range(len(boxes_data)):
            class_id = int(class_ids[i])
            label = class_names[class_id]
            confidence = confidences[i]
            bbox = boxes_data[i]

            predictions.append({
                'name': label,
                'confidence': confidence,
                'bbox': bbox
            })

        print(f"Predictions: {predictions}")

        # Filter out 'Bullseye' and process predictions
        filtered_preds = [pred for pred in predictions if pred['name'] != '0' and pred['confidence'] > 0.5]

        if not filtered_preds:
            print("No valid predictions after filtering.")
            return 'NA'

        # Select the prediction with the highest confidence
        # best_pred = max(filtered_preds, key=lambda x: x['confidence'])
        # print(f"Best Prediction: {best_pred}")

        # Map label to ID
        # name_to_id = {
        #     "NA": 'NA',
        #     "0": 0,
        #     "11": 11,
        #     "12": 12,
        #     "13": 13,
        #     "14": 14,
        #     "15": 15,
        #     "16": 16,
        #     "17": 17,
        #     "18": 18,
        #     "19": 19,
        #     "20": 20,
        #     "21": 21,
        #     "22": 22,
        #     "23": 23,
        #     "24": 24,
        #     "25": 25,
        #     "26": 26,
        #     "27": 27,
        #     "28": 28,
        #     "29": 29,
        #     "30": 30,
        #     "31": 31,
        #     "32": 32,
        #     "33": 33,
        #     "34": 34,
        #     "35": 35,
        #     "36": 36,
        #     "37": 37,
        #     "38": 38,
        #     "39": 39,
        #     "36": 36,
        #     "40": 40
        # }
        # Select the prediction with the highest confidence
        best_pred = max(filtered_preds, key=lambda x: x['confidence'])
        print(f"Best Prediction: {best_pred}")
        
        # Draw bounding box on the image
        x1, y1, x2, y2 = best_pred['bbox']
        #draw_own_bbox(np.array(img), x1, y1, x2, y2, best_pred['name'])
        # Ensure image_id is a string
        #image_id = best_pred['name']
        image_id = str(best_pred['name'])  # Convert to string explicitly
        print(f"Final result: {image_id}")  # If this prints, the error is resolved
        return image_id

        #image_id = str(name_to_id.get(str(best_pred['name']), 'NA'))
        # image_id = best_pred['name']
        # print(f"Final result: {image_id}")
        # return image_id

    except Exception as e:
        print(f"Error during prediction: {e}")
        return 'NA'


def predict_image_week_9(image, model):
    # Load the image
    img = Image.open(os.path.join('images', image))
    # Run inference
    results = model(img)
    # Save the results
    results.save('runs')
    # Convert the results to a dataframe
    first_result = results[0]
    first_result.save('runs')
    df_results = first_result.pandas().xyxy[0]

    #df_results = results.pandas().xyxy[0]
    # Calculate the height and width of the bounding box and the area of the bounding box
    df_results['bboxHt'] = df_results['ymax'] - df_results['ymin']
    df_results['bboxWt'] = df_results['xmax'] - df_results['xmin']
    df_results['bboxArea'] = df_results['bboxHt'] * df_results['bboxWt']

    # Label with largest bbox height will be last
    df_results = df_results.sort_values('bboxArea', ascending=False)
    pred_list = df_results 
    pred = 'NA'
    # If prediction list is not empty
    if pred_list.size != 0:
        # Go through the predictions, and choose the first one with confidence > 0.5
        for _, row in pred_list.iterrows():
            if row['name'] != '0' and row['confidence'] > 0.5:
                pred = row    
                break

        # Draw the bounding box on the image 
        if not isinstance(pred,str):
            draw_own_bbox(np.array(img), pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'], pred['name'])
        
    # Dictionary is shorter as only two symbols, left and right are needed
    name_to_id = {
        "NA": 'NA',
        "Bullseye": 10,
        "Right": 38,
        "Left": 39,
        "Right Arrow": 38,
        "Left Arrow": 39,
    }
    # Return the image id
    if not isinstance(pred,str):
        image_id = str(name_to_id[pred['name']])
    else:
        image_id = 'NA'
    return image_id


def stitch_image():
    """
    Stitches the images in the folder together and saves it into runs/stitched folder
    """
    # Initialize path to save stitched image
    imgFolder = 'runs'
    stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

    # Find all files that ends with ".jpg" (this won't match the stitched images as we name them ".jpeg")
    imgPaths = glob.glob(os.path.join(imgFolder+"/detect/*/", "*.jpg"))
    # Open all images
    images = [Image.open(x) for x in imgPaths]
    # Get the width and height of each image
    width, height = zip(*(i.size for i in images))
    # Calculate the total width and max height of the stitched image, as we are stitching horizontally
    total_width = sum(width)
    max_height = max(height)
    stitchedImg = Image.new('RGB', (total_width, max_height))
    x_offset = 0

    # Stitch the images together
    for im in images:
        stitchedImg.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    # Save the stitched image to the path
    stitchedImg.save(stitchedPath)

    # Move original images to "originals" subdirectory
    for img in imgPaths:
        shutil.move(img, os.path.join(
            "runs", "originals", os.path.basename(img)))

    return stitchedImg

def stitch_image_own():
    """
    Stitches the images in the folder together and saves it into own_results folder

    Basically similar to stitch_image() but with different folder names and slightly different drawing of bounding boxes and text
    """
    imgFolder = 'own_results'
    stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

    imgPaths = glob.glob(os.path.join(imgFolder+"/annotated_image_*.jpg"))
    imgTimestamps = [imgPath.split("_")[-1][:-4] for imgPath in imgPaths]
    
    sortedByTimeStampImages = sorted(zip(imgPaths, imgTimestamps), key=lambda x: x[1])

    images = [Image.open(x[0]) for x in sortedByTimeStampImages]
    width, height = zip(*(i.size for i in images))
    total_width = sum(width)
    max_height = max(height)
    stitchedImg = Image.new('RGB', (total_width, max_height))
    x_offset = 0

    for im in images:
        stitchedImg.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    stitchedImg.save(stitchedPath)

    return stitchedImg

