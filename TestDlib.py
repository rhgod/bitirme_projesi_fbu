import cv2
import dlib
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
cap = cv2.VideoCapture("./videos/friends2.mp4")
# cap = cv2.VideoCapture("./videos/sad1.mp4")

while True:
    ret, frame = cap.read()
    # yüz tanımayı daha hızlı yapmak için frame boyutunu 1/4 boyutuna küçültüyoruz
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    if not ret:
        break
    # dlib ile yüz tespiti yapıyoruz
    faceDetector = dlib.get_frontal_face_detector()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detectedFaces = faceDetector(grayFrame, 1)

    # her yüz için dlib ile yüzün koordinatlarını alıp ve çerçeve içine alıyoruz
    for face in detectedFaces:
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        grayFrameImg = grayFrame[face.top():face.bottom(), face.left():face.right()]
        croppedImg = np.expand_dims(np.expand_dims(cv2.resize(grayFrameImg, (48, 48)), -1), 0)

        # modele göre duygu tahmini yapıyoruz
        emotionPrediction = emotionModel.predict(croppedImg)
        maxIndex = int(np.argmax(emotionPrediction))
        cv2.putText(frame, emotionDictionary[maxIndex], (face.left()+5, face.bottom()+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_4)

        # duygu tahminini frame üzerine yazdırıyoruz
        cv2.putText(frame, str(emotionPrediction[0][maxIndex]), (face.left()+5, face.bottom()+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_4)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
