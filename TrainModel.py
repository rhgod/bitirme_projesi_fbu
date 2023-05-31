import cv2
import time
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator 

# ImageDataGenerator ile train ve test verilerini yükleyip rescale ediyoruz
trainDataGenerator = ImageDataGenerator(rescale=1./255)
validationDataGenerator = ImageDataGenerator(rescale=1./255)

# Train verilerini yükleyip preprocess ediyoruz
trainGenerator = trainDataGenerator.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical'
)

# Test verilerini yükleyip preprocess ediyoruz
validationGenerator = validationDataGenerator.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical'
)

# Modeli oluşturuyoruz
emotionModel = Sequential()

emotionModel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotionModel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotionModel.add(MaxPooling2D(pool_size=(2, 2)))
emotionModel.add(Dropout(0.25))

emotionModel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotionModel.add(MaxPooling2D(pool_size=(2, 2)))
emotionModel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotionModel.add(MaxPooling2D(pool_size=(2, 2)))
emotionModel.add(Dropout(0.25))

emotionModel.add(Flatten())
emotionModel.add(Dense(1024, activation='relu'))
emotionModel.add(Dropout(0.5))
emotionModel.add(Dense(7, activation='softmax'))

# False ise CPU üzerinde eğitim yapar True ise GPU üzerinde eğitim yapar.
cv2.ocl.setUseOpenCL(False)

# Modeli derliyoruz
emotionModel.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

startTime = time.time()

# Modeli eğitiyoruz
emotionModelInfo = emotionModel.fit_generator(
        trainGenerator,
        steps_per_epoch=28709 // 64, # = 448 steps per epoch (28709 images in train folder)
        epochs=50,
        validation_data=validationGenerator,
        validation_steps=7178 // 32 # = 224 steps per epoch (7178 images in test folder)
) 

endTime = time.time()

# Modeli json formatında kaydediyoruz
modelJson = emotionModel.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(modelJson)

# Eğiitilmiş modelin ağırlıklarını h5 formatında kaydediyoruz.
emotionModel.save_weights('emotion_model.h5')

# Modelin özetini yazdırıyoruz
print(emotionModel.summary())

totalTime = endTime - startTime

# Toplam eğitim süresini yazdırıyoruz
print("Time taken to train the model with 50 epochs: ", totalTime, " seconds", "(", totalTime/60, " minutes", "(", totalTime/3600, " hours", ")")


