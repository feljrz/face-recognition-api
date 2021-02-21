from flask import Flask, Response, render_template, request
import cv2 
import queue, threading, time



app = Flask(__name__)



@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    """Video streaming generator function."""
    # global cap
    cap = cv2.VideoCapture(0)

    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else: 
            break
        

@app.route('/video_feed')
def video_feed():
    # if request.method == 'GET':
    #     generator =
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture', methods=['GET', 'POST', 'PUT'])
def capture_frame():
    global cap
    if ret and request.method == 'POST':
        img = cv2.resize((0,0), fx=0.5, fy=0.5)
        cv2.imwrite('frame.jpg', img)
    
 

if __name__ == "__main__":
    app.run(debug=True)