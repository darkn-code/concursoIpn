#!/usr/bin/python3
import time
import datetime
import os
import jetson.inference
import jetson.utils
import pandas as pd

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("/dev/video0")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file
try:
    tabla = pd.read_csv('./CSV/registro.csv')
    if not tabla.empty:
        tabla.drop(["Unnamed: 0"],axis=1,inplace=True)
    n = len(tabla)
except:
    tabla = pd.DataFrame({"Nombre":[],"Fecha":[],"Hora":[]})
    n = 0
while display.IsStreaming():
    img = camera.Capture()
    detections = net.Detect(img)
    for detection in detections:
        print(detection.ClassID)
        if detection.ClassID == 1:
            fechaYhoraactual = datetime.datetime.now()
            fecha = fechaYhoraactual.strftime("%d/%m/%Y")
            hora = fechaYhoraactual.strftime("%H:%M:%S")
            nueva_fila={"Nombre":"persona"+str(n),"Fecha":fecha,"Hora":hora}
            tabla = tabla.append(nueva_fila,ignore_index=True)
            tabla.to_csv("./CSV/registro.csv")
            print(tabla)
            jetson.utils.saveImage('./imagenes/personas'+str(n)+'.png',img)
            time.sleep(5)
            n+=1
            os.system('git commit -am "Registro"')
            os.system('git push origin master')

    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
