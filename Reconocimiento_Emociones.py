import cv2


# Cargar los modelos de detección facial y de emociones
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


# Definir los nombres de las emociones
emotions = {
    0: 'Felicidad',
    1: 'Tristeza',
    2: 'Sorpresa',
    3: 'Miedo',
    4: 'Asco',
    5: 'Enojo',
    6: 'Neutral'
}

# Iniciar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar un marco de video
    ret, frame = cap.read()
    
    # Convertir el marco a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar las caras en el marco de video
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Para cada cara detectada
    for (x, y, w, h) in faces:
        # Extraer la región de interés de la cara
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detectar las sonrisas en la región de interés de la cara
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        
        # Clasificar la emoción de la cara
        if len(smiles) == 0:
            # Si no se detecta ninguna sonrisa, la emoción es "Neutral"
            emotion = emotions[6]
        else:
            # Si se detecta una sonrisa, se clasifica la emoción en base a la intensidad de la sonrisa
            for (ex, ey, ew, eh) in smiles:
                smile_area = (ex + ew) * (ey + eh)
                face_area = w * h
                smile_intensity = smile_area / face_area
                print(smile_intensity)
                if smile_intensity < 0.5:
                    emotion = emotions[0]  # Felicidad
                 
                elif smile_intensity < 0.7:
                    emotion = emotions[2]  # Sorpresa
      
                elif smile_intensity < 0.9:
                    emotion = emotions[3]  # Miedo
              
                elif smile_intensity < 0.12:
                    emotion = emotions[5]  # Enojo
                    
                elif smile_intensity < 0.4:
                    emotion = emotions[1]  # Tristeza
                   
                else:
                    emotion = emotions[4]  # Asco
            
        # Dibujar un rectángulo alrededor de la cara y mostrar la emoción en la pantalla
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Mostrar el marco de video en la pantalla
    cv2.imshow('Video', frame)
    
    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Detener la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
