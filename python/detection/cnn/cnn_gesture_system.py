import cv2
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


class CnnGestureSystem:

    def __init__(self):
        self._detecting = False

    # Test for VGG16 CNN model (https://www.geeksforgeeks.org/vgg-16-cnn-model/)
    @staticmethod
    def create_model1():
        dataset_generator = ImageDataGenerator(validation_split=0.2)
        train_data = dataset_generator.flow_from_directory(directory="../../../dataset", target_size=(224, 224),
                                                           subset='training')
        test_data = dataset_generator.flow_from_directory(directory="../../../dataset", target_size=(224, 224),
                                                          subset='validation')

        model = Sequential()
        model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=8, activation="softmax"))

        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
        model.summary()

        checkpoint = ModelCheckpoint("../../../models/model1.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto')

        early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

        hist = model.fit_generator(steps_per_epoch=100, generator=train_data, validation_data=test_data,
                                   validation_steps=10,
                                   epochs=100, callbacks=[checkpoint, early])

        plt.plot(hist.history["accuracy"])
        plt.plot(hist.history['val_accuracy'])
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title("model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
        plt.show()

    @staticmethod
    def create_model2():
        dataset_generator = ImageDataGenerator(validation_split=0.2)
        train_data = dataset_generator.flow_from_directory(directory="../../../dataset", target_size=(224, 224),
                                                           subset='training')
        test_data = dataset_generator.flow_from_directory(directory="../../../dataset", target_size=(224, 224),
                                                          subset='validation')

        model = Sequential()
        model.add(Conv2D(input_shape=(224, 224, 3), filters=16, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())

        model.add(Dense(units=48, activation="relu"))
        model.add(Dense(units=8, activation="softmax"))

        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
        model.summary()

        checkpoint = ModelCheckpoint("../../../models/model2.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto')

        early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

        hist = model.fit_generator(steps_per_epoch=100, generator=train_data, validation_data=test_data,
                                   validation_steps=10,
                                   epochs=100, callbacks=[checkpoint, early])

        plt.plot(hist.history["accuracy"])
        plt.plot(hist.history['val_accuracy'])
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title("model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
        plt.show()

    @staticmethod
    def create_model3():
        dataset_generator = ImageDataGenerator(validation_split=0.2)
        train_data = dataset_generator.flow_from_directory(directory="../../../dataset", target_size=(224, 224),
                                                           subset='training')
        test_data = dataset_generator.flow_from_directory(directory="../../../dataset", target_size=(224, 224),
                                                          subset='validation')

        model = Sequential()
        model.add(Conv2D(input_shape=(224, 224, 3), filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())

        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=8, activation="softmax"))

        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
        model.summary()

        checkpoint = ModelCheckpoint("../../../models/model3.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto')

        early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

        hist = model.fit_generator(steps_per_epoch=100, generator=train_data, validation_data=test_data,
                                   validation_steps=10,
                                   epochs=100, callbacks=[checkpoint, early])

        plt.plot(hist.history["accuracy"])
        plt.plot(hist.history['val_accuracy'])
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title("model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
        plt.show()

    @staticmethod
    def create_model4():
        dataset_generator = ImageDataGenerator(validation_split=0.2)
        train_data = dataset_generator.flow_from_directory(directory="../../dataset", target_size=(224, 224),
                                                           subset='training')
        test_data = dataset_generator.flow_from_directory(directory="../../dataset", target_size=(224, 224),
                                                          subset='validation')

        model = Sequential()
        model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())

        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=8, activation="softmax"))

        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
        model.summary()

        checkpoint = ModelCheckpoint("../../models/model4.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto')

        early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

        hist = model.fit_generator(steps_per_epoch=100, generator=train_data, validation_data=test_data,
                                   validation_steps=10,
                                   epochs=100, callbacks=[checkpoint, early])

        plt.plot(hist.history["accuracy"])
        plt.plot(hist.history['val_accuracy'])
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title("model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
        plt.show()

    @staticmethod
    def create_model5():
        dataset_generator = ImageDataGenerator(validation_split=0.2)
        train_data = dataset_generator.flow_from_directory(directory="../../../dataset", target_size=(224, 224),
                                                           subset='training')
        test_data = dataset_generator.flow_from_directory(directory="../../../dataset", target_size=(224, 224),
                                                          subset='validation')

        model = Sequential()
        model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())

        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=8, activation="softmax"))

        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
        model.summary()

        checkpoint = ModelCheckpoint("../../../models/model5.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto')

        early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

        hist = model.fit_generator(steps_per_epoch=100, generator=train_data, validation_data=test_data,
                                   validation_steps=10,
                                   epochs=100, callbacks=[checkpoint, early])

        plt.plot(hist.history["accuracy"])
        plt.plot(hist.history['val_accuracy'])
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title("model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
        plt.show()

    @staticmethod
    def load_model(path):
        return keras.models.load_model(path)

    @staticmethod
    def create_detection_image(frame):
        # Gray scale
        '''image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.bitwise_not(image)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        return image'''
        # Another method
        # Extract red color channel (because the hand color is more red than the background).
        gray = frame[:, :, 2]
        # Apply binary threshold using automatically selected threshold (using cv2.THRESH_OTSU parameter).
        ret, thresh_gray = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
        # Use "opening" morphological operation for clearing some small dots (noise)
        thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        # Use "closing" morphological operation for closing small gaps
        thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        image = cv2.bitwise_not(thresh_gray)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.array(image)
        return image

    def detect(self, model):

        if isinstance(model, str):
            model = CnnGestureSystem.load_model(model)

        camera = cv2.VideoCapture(0)

        while self._detecting:
            ret, frame = camera.read()
            image1 = CnnGestureSystem.create_detection_image(frame)
            image = np.array(image1)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)

            # classify the image
            preds = model.predict(image, verbose=0)
            labels = ["apri", "avanti", "chiudi", "destra", "indietro", "sinistra", "sopra", "sotto"]
            predictions = []
            for i in range(0, 8):
                predictions.append([labels[i], preds[0][i]])
            prediction_index = np.argmax(preds[0])
            (label, accuracy) = predictions[prediction_index]

            # Impress Label and show the image
            print(predictions)
            print("Label: {}".format(label))
            cv2.putText(image1, "Label: {}".format(label), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Classification", image1)
            cv2.waitKey(100)

        camera.release()
        cv2.destroyAllWindows()

    def start_detection(self, model):
        self._detecting = True
        self.detect(model)

    def stop_detection(self):
        self._detecting = False
