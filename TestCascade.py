import cv2
import numpy as np
from keras.models import model_from_json

# duygu tahminlerini tutan dictionary
emotionDictionary = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# json dosyasını yükle ve modeli oluştur
jsonFile = open('model/emotion_model.json', 'r')
loadedModelJson = jsonFile.read()
jsonFile.close()
emotionModel = model_from_json(loadedModelJson)

# ağırlıkları yükle (weights) ve modeli derle (compile)
emotionModel.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# webcam kullanmak için
#cap = cv2.VideoCapture(0)

# video path kullanmak için
cap = cv2.VideoCapture("./videos/friends1.mp4")
# cap = cv2.VideoCapture("./videos/sad1.mp4")

while True:
    ret, frame = cap.read()
    # yüz tanımayı daha hızlı yapmak için frame boyutunu 1/4 boyutuna küçültüyoruz
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    if not ret:
        break
    # yüz tespiti için cascade classifier kullanıyoruz
    faceDetector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # görüntüdeki yüzleri tespit etmek için detectMultiScale kullanıyoruz
    detectedFaces = faceDetector.detectMultiScale(grayFrame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in detectedFaces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
        grayFrameImg = grayFrame[y:y + h, x:x + w]
        croppedImg = np.expand_dims(np.expand_dims(cv2.resize(grayFrameImg, (48, 48)), -1), 0)

        # tahmin
        emotionPrediction = emotionModel.predict(croppedImg)
        maxIndex = int(np.argmax(emotionPrediction))
        cv2.putText(frame, emotionDictionary[maxIndex], (x+5, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_4)

        # tahmin olasılıklarını yazdır
        cv2.putText(frame, str(emotionPrediction[0][maxIndex]), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_4)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
