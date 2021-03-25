import PySimpleGUI as sg
import cv2
import numpy as np
import threads_cam_copy



def main():

    sg.theme('Black')

    # define the window layout
    layout = [[sg.Text('OpenCV Demo', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image')],
              [sg.Button('Record', size=(10, 1), font='Helvetica 14'),
               sg.Button('Stop', size=(10, 1), font='Any 14'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14'), ]]

    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration',
                       layout, location=(800, 400))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = threads_cam_copy.VideoCapture(0)
    recording = False

    while True:
        event, values = window.read(timeout=2)
        if event == 'Sair' or event == sg.WIN_CLOSED:
            return

        elif event == 'Record':
            recording = True

        elif event == 'Stop':
            recording = False
            img = np.full((480, 640), 255)
            # this is faster, shorter and needs less includes
            imgbytes = cv2.imencode('.png', img)[1].tobytes()
            window['image'].update(data=imgbytes)

        if recording:
            frame = cap.read()
            imgbytes = threads_cam_copy.cam_gen(frame)
            # frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
            # imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)


if __name__ == "__main__":
    main()