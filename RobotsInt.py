import cv2
from keras.models import load_model
import numpy as np

# Cargar el modelo entrenado
model = load_model('model.h5')

# Definir las etiquetas de las emociones
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Crear un objeto de captura de video
cap = cv2.VideoCapture(0)

while True:
    # Leer un fotograma de la captura de video
    ret, frame = cap.read()

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar las caras en el fotograma
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)

    # Procesar cada cara detectada
    for (x, y, w, h) in faces:
        # Extraer la región de interés (ROI) correspondiente a la cara
        roi_gray = gray[y:y+h, x:x+w]

        # Redimensionar la ROI a 48x48 (tamaño esperado por el modelo)
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalizar la intensidad de los píxeles en la ROI
        roi_gray = roi_gray / 255.0

        # Convertir la ROI a un arreglo de numpy
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

        # Hacer la predicción con el modelo
        preds = model.predict(roi_gray)[0]
        emotion = EMOTIONS[np.argmax(preds)]

        # Dibujar un rectángulo alrededor de la cara y escribir la emoción detectada
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el fotograma procesado
    cv2.imshow('Video', frame)

    # Esperar a que se presione la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
