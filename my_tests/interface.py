import random
import threading
import os
import PySimpleGUI as sg
import threads_cam_copy

sg.theme('DarkAmber')

def cadastrar_usuario():
    sg.theme('DarkBlue')
    layout = [
        [sg.Button('Voltar', size=(12, 1))],
        [sg.Text('Digite o Nome:'), sg.InputText(size=(100,2), key='nome')],
        [sg.Text('Digite a Matrícula:'), sg.InputText(size=(100,2), key="matricula")],
        [sg.Text('POSICIONE SEU ROSTO EM FRENTE A CAMÊRA', justification='center', size=(100, 1))],
        [sg.Output(size=(100, 30))],
        [sg.Button('TIRAR SCREENSHOT', size=(25, 1))]

    ]
    window = sg.Window('Cadastrar', layout, modal=True, size=(800, 700), element_justification='c')
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        if event == 'Voltar':
            window.close()
        if event == 'TIRAR SCREENSHOT':
            print("SCREENSHOT")
        #   threads_cam_copy.cam_gen()

    window.close()

def main():
    sg.theme('DarkBlue')
    layout = [[sg.Text('Portaria', size=(40, 1), justification='center', font='Helvetica 20')],
          [sg.ReadButton('Sair', size=(10, 1), font='Helvetica 14'),
           sg.RButton('Iniciar', size=(10,1), font='Helvetica 14')],
           [sg.Image(filename='', key='image')],
           [sg.RButton('Cadastrar', size=(10,1), font='Helvetica 14')]]

    window = sg.Window('Reconhecimento Facial', layout, modal=True, size=(700, 500), element_justification='c')

    cap = threads_cam_copy.VideoCapture(0)
    record = False
    while True:
        event, values = window.read(timeout=20)
        if event == 'Sair' or event == sg.WIN_CLOSED:
            cap.release()
            return
        if event == 'Iniciar':
            record = True

        if record:
            frame = cap.read()
            imgbytes = threads_cam_copy.cam_gen(frame)
            window['image'].update(data=imgbytes)
      


        # if event == 'Cadastrar Usuário':
        #     pass
            # cadastrar_usuario()
    window.close()

if __name__ == "__main__":
    cam_gen_thread = threading.Thread(target=main)
    cam_gen_thread.daemon = True
    cam_gen_thread.start()
    cam_gen_thread.join()
