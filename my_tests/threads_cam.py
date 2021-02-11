from flask import Flask, Response, render_template, request, make_response
import queue, threading, time
import json
import numpy as np
import face_recognition as fr
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import fr_api_wrapper as fr_wp


class VideoCapture():
    def __init__(self, webcam_name):
        self.lock = threading.Lock()
        self.webcam_name = webcam_name
        self.q = queue.Queue()
        self.cap = cv2.VideoCapture(self.webcam_name)
        cap_thread = threading.Thread(target=self.buffer)
        cap_thread.deamon = True
        cap_thread.start() # não pode iniciar automáticamente, será por evento

    def buffer(self):
        self.lock.acquire()
        while(True):
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait() #Descartar frames não processados
                except queue.Empty:
                    pass
            self.q.put(frame)
        self.lock.release()
    
    def read(self):
        return self.q.get()
    
#Deve receber o nome e por padrao NONE
def cam_gen():
    capture = VideoCapture(0)
    while(True):
        im = capture.read()
        small_frame = cv2.resize(im, (0, 0), fx=0.7, fy=0.7)
        face_location, face_encoding = decode(small_frame)
        if len(face_location) == 1:
            try:
                draw_thread = threading.Thread(target=draw_rectangle)
                draw_thread._args = (small_frame, face_location, "Felipe")
                draw_thread.start()
            except:
                pass
            # draw_rectangle(small_frame, "Felipe", face_location)
        ret_enc, buffer = cv2.imencode('.jpg', small_frame)
        encode_frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + encode_frame + b'\r\n')

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    # global cap
    # threading.Thread(target=cam_gen, args=(cap,)
    return Response(cam_gen(),
             mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/v1/screenshot', methods=['POST'])
def screenshot():
    loc_photo_screenshot = "capturas/frame1.jpg" #Lembrar de susbstituir por NomeUsuario%NumeroFoto
    loc_csv_screenshot = "capturas/SCREENSHOT.csv"
    if request.method == 'POST':
        save_type = request.args.get("type")
        capture = VideoCapture(0)
        while(True):
            frame = capture.read()
            time.sleep(0)
            break
        capture.cap.release()

        #não é seguro
        #lento
        if save_type == "image":
            try:
                saved = cv2.imwrite(loc_photo_screenshot, frame)
                df = pd.DataFrame({"Frame": [frame]})
                return make_response(json.dumps(df.to_json(orient='records')))
                # return make_response(json.dumps({"Saved": saved}))

            except OSError as err:
                return make_response(json.dumps({"Error": err}))


        face_location, face_encoding = decode(frame)
        if face_location is not None:
            try:
                mapped = {"Face Location": face_location, "Face Encoding": face_encoding}    
                if save_type == "both":
                    df = pd.DataFrame(mapped)
                    saved = cv2.imwrite(loc_photo_screenshot, frame)
                    df.to_csv(loc_csv_screenshot, index=False)
                    return make_response(df.to_json(orient='records')) #Adicionar photo?
                else:
                    df = pd.DataFrame(mapped)
                    df.to_csv(loc_csv_screenshot, index=False)
                    return make_response(df.to_json(orient='records'))

            except OSError as err:
                return make_response(json.dumps({"Error": err}))
        
    


def decode(frame = None, im_loc=None):
    if im_loc is not None:
        frame = cv2.imread(im_loc)
    face_location = fr.face_locations(frame)
    face_encoding = fr.face_encodings(frame, face_location)
    return (face_location, face_encoding)


def draw_rectangle(frame, faces_locations, name):
    for face_location in faces_locations:
        face_location = list(face_location)
        face_location_big = list(map(lambda x: int(x*1.05), face_location))


        top, right, bottom, left = face_location_big
        print(f'{top}+" "+{right}+" "+{bottom}+" "+{left}')
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left, bottom), font, 0.7, (255, 255, 255), 1)
        time.sleep(0)
    

def recognize()

if __name__ == "__main__":
    # threading.Thread(target=app.run(debug=True)).start()
    app.run(debug=True)
    loc_save_screenshot = "capturas/frame.jpg"
    # screenshot(loc_save_screenshot)
    # decode(loc_save_screenshot)


























# while(True):
#     time.sleep(0.0)
#     frame = capture.read()
#     cv2.imshow("frame", frame)
#     if chr(cv2.waitKey(1)&255) == 'q':
#         break






# import cv2, queue, threading, time

# # bufferless VideoCapture
# class VideoCapture:

#   def __init__(self, name):
#     self.cap = cv2.VideoCapture(name)
#     self.q = queue.Queue()
#     t = threading.Thread(target=self._reader)
#     t.daemon = True
#     t.start()

#   # read frames as soon as they are available, keeping only most recent one
#   def _reader(self):
#     while True:
#       ret, frame = self.cap.read()
#       if not ret:
#         break
#       if not self.q.empty():
#         try:
#           self.q.get_nowait()   # discard previous (unprocessed) frame
#         except queue.Empty:
#           pass
#       self.q.put(frame)

#   def read(self):
#     return self.q.get()

# cap = VideoCapture(0)
# while True:
#   time.sleep(.5)   # simulate time between events
#   frame = cap.read()
#   cv2.imshow("frame", frame)
#   if chr(cv2.waitKey(1)&255) == 'q':
#     break