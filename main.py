from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

# Carga los modelos una sola vez al inicio del programa
model = YOLO('best.pt')
reader = easyocr.Reader(['en'])

# Inicia la aplicación FastAPI
app = FastAPI()

# Define un endpoint para la predicción 
@app.post("/detectar_patente/")
async def detectar_patente_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(image)
    
    patente_detectada = "No se detecto ninguna patente"

    for r in results:
        if len(r.boxes) > 0:
            mejor_conf_idx = np.argmax(r.boxes.conf.cpu().numpy())
            mejor_caja = r.boxes.xyxy[mejor_conf_idx]
            
            xmin, ymin, xmax, ymax = map(int, mejor_caja)
            patente_recortada = image[ymin:ymax, xmin:xmax]

            ocr_results = reader.readtext(patente_recortada)

            if ocr_results:
                texto_leido = ocr_results[0][1]
                patente_detectada = texto_leido

    return {"patente": patente_detectada}