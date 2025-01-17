from flask import Flask, Response, render_template, request, make_response
import queue, threading, time
import math
import json
import numpy as np
import face_recognition as fr
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import fr_api_wrapper as fr_api


def face_size(face):
    top, right, bottom, left = face
    distance = math.sqrt((top-bottom)**2 + (right - left)**2)
    # print(f"DISTANCE: {distance}")
    return distance

"""Removing encoded faces on frames that has more than one face"""
def remove_background_faces(raw_location, raw_encoding):
    n_faces = len(raw_location)
    result = zip(raw_location, raw_encoding)
    faces_dict = dict(result)
    sorted_faces = dict()

    if n_faces == 1:
        return raw_location, raw_encoding
    else:
        #Ordering a dict thought the proximity face
        for face_location in sorted(faces_dict, key= lambda x: face_size(x), reverse=True):
            sorted_faces.update({face_location: faces_dict[face_location]})

        closest_face = next(iter(sorted_faces.items()))
        cl_face_loc, cl_face_encode = closest_face
        return [cl_face_loc], [cl_face_encode]

def recognize(frame_encoded):
    global model
    neighbors = model.kneighbors(frame_encoded, n_neighbors = 4)

    # ? IMPORTANT HERE (INITIALLY) I CONSIDER THAT EACH PERSON ON ENTIRE DATASET HAS MORE THAN ONE PICTURE
    # 0.6 Tolerance is just to use dataset faces
    # name, wrong_aws_count = fr_api.get_label(frame_encoded, neighbors[1], 0.5) 
    # if name[0] == None:
    #     name = model.predict(frame_encoded)

    name = model.predict(frame_encoded)
   
    return name



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

    
def cam_gen(name = None):
    capture = VideoCapture(0)
    while(True):
        im = capture.read()
        small_frame = cv2.resize(im, (0, 0), fx=0.6, fy=0.6)
        face_location, face_encoding = decode(small_frame)
     
        if len(face_location) >= 1:
            
            #Will be used to verify veracity
            is_real = True if face_size(face_location[0]) > 100 else False
            full_name = True
        
            if len(face_location) > 1:
                face_location, face_encoding = remove_background_faces(face_location, face_encoding)
            
            try:
                name = recognize(face_encoding)[0]
                # print("INIT DRAW")
                draw_thread = threading.Thread(target=draw_rectangle)
                draw_thread._args = (small_frame, face_location, name, is_real, full_name)
                draw_thread.start()
                draw_thread.join()
            except:
                pass
            
        ret_enc, buffer = cv2.imencode('.jpg', small_frame)
        encode_frame = buffer.tobytes()
        # print("END GEN")
        # print("__________")
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


def draw_rectangle(frame, faces_locations, name, alive, full_name=None):
    for face_location in faces_locations:
        face_location = list(face_location)
        face_location_big = list(map(lambda x: int(x*1.05), face_location))

        show_name = ' '.join(name.split('_')) if full_name else name.split('_')[0]
        color = (0, 200, 0) if alive else (0, 0, 255)


        top, right, bottom, left = face_location_big
        # print(f'{top}+" "+{right}+" "+{bottom}+" "+{left}')
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, show_name, (left, bottom), font, 0.7, (255, 255, 255), 1)
        # print("END DRAW")

        # time.sleep(0)
    

# #Ativado por evento (click)
# def recognize(face_encoding, model):
    
#     cam_gen()

# abspath = "/home/felipe/Documents/Alianca/image-explorer/my_tests"
abspath = os.path.abspath(os.getcwd()) #certo


model_save_path = abspath + "/bkp/knn_model.clf"
train_dir = abspath + "/archive/lfw-deepfunneled"
df_save_path = abspath + "/bkp/dataset_example.pkl"
loc_save_screenshot = abspath + "/capturas/frame.jpg"

model = fr_api.load_binary(model_save_path)

if __name__ == "__main__":
    retrain_model = False
    if retrain_model:
        print("Start retrain")
        fr_api.first_train(train_dir, model_save_path, df_save_path)
        print("End of retrain")

    app.run(debug=True)
