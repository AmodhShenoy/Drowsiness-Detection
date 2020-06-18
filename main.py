from flask import Flask, render_template, Response,redirect
from camera import DrowsyDetector
import time
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('drowsy.html')

def gen():
    camera = DrowsyDetector()
    isClosed = False
    start_time = time.time()
    mustStop = False
    while True:
        status,frame = camera.get_frame()
        if status=='close' and not isClosed:
            start_time = time.time()
        elif status=='open' and isClosed:
            start_time = 0

        if status=='close' and time.time()-start_time>5:
            mustStop = True
        isClosed = status=='close'

        if mustStop:
            print("ALERT!!!")
            return redirect('/alert')
            image = cv2.imread('static/images/alert.jpg')
            # cv2.imshow(image)
            ret,jpeg = cv2.imencode('.jpg', image)
            return (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alert')
def alert():
    return render_template('alert.html')

if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)
