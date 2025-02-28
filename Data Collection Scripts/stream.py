from flask import Flask, Response
import picamera
import io
import time

app = Flask(__name__)

def generate_frames():
    """Captures frames from the Raspberry Pi camera and streams them."""
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.framerate = 30
        camera.start_preview()
        time.sleep(2)  # Warm-up time

        stream = io.BytesIO()
        for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
            stream.seek(0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + stream.read() + b'\r\n')
            stream.seek(0)
            stream.truncate()

@app.route('/video_feed')
def video_feed():
    """Route to stream the video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
