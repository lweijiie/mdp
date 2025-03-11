import time
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import *


app = Flask(__name__)
CORS(app)
model = load_model()
#model = None
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if not os.path.exists('runs'):
    os.makedirs('runs')
@app.route('/status', methods=['GET'])
def status():
    """
    This is a health check endpoint to check if the server is running
    :return: a json object with a key "result" and value "ok"
    """
    return jsonify({"result": "ok"})

@app.route('/image', methods=['POST'])
def image_predict():
    """
    This is the main endpoint for the image prediction algorithm
    :return: a json object with a key "result" and value a dictionary with keys "obstacle_id" and "image_id"
    """
    file = request.files['file']
    filename = file.filename
    file.save(os.path.join('images', filename))
    # filename format: "<timestamp>_<obstacle_id>_<signal>.jpeg"
    constituents = file.filename.split("_")
    obstacle_id = constituents[1] if len(constituents) > 1 else "default"

    if len(constituents) > 1:
        obstacle_id = constituents[1]
    else:
        obstacle_id = "default"  # or handle the case differently
    constituents = file.filename.split("_")
    obstacle_id = constituents[1]

    ## Week 8 ## 
    signal = constituents[2].strip(".jpg")
    image_id = predict_image(filename, model, signal)

    ## Week 9 ## 
    # We don't need to pass in the signal anymore
    #image_id = predict_image_week_9(filename,model)

    # Return the obstacle_id and image_id
    #image_id, predictions_json = predict_image(filename, model, signal)

    # Ensure image_id is returned as the predicted class ID
    #try:
       # image_id = predictions_json[0]['name']  # Set image_id to the predicted class name directly
   # except Exception:
       # image_id = "NA"


    result = {
        "obstacle_id": obstacle_id,
        "image_id": image_id,
    }

    print("Prediction Result (Console):", result)

    # Log details to the console
    print("Image ID:", image_id)
    print("Obstacle ID:", obstacle_id)

    return jsonify(result)

@app.route('/stitch', methods=['GET'])
def stitch():
    """
    This is the main endpoint for the stitching command. Stitches the images using two different functions, in effect creating two stitches, just for redundancy purposes
    """
    img = stitch_image()
    img.show()
    img2 = stitch_image_own()
    img2.show()
    return jsonify({"result": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
