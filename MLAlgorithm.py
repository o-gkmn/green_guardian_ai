import csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import joblib

class MLAlgorithmTrain:
    data = []
    aso_ids = []
    audio_files_with_labels = []

    def __init__(self) -> None:
        pass

    def __load_audio_file(self, file_path):
        audio, sr = librosa.load("data\\train_data\\"+file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

        return np.mean(mfccs, axis=1)

    def __set_audio_files_with_labels(self):
        df = pd.read_csv('meta\\train.csv')
        self.audio_files_with_labels = df.values.tolist()

    def __load_multiple_audio_file(self):
        for file_path, label, aso_id, manually_verified, noisy_small in self.audio_files_with_labels:
            mfccs = self.__load_audio_file(file_path)
            self.data.append(mfccs)
            self.aso_ids.append(self.__decide_aso_id_equilevent_int(aso_id))

    def __convert_data_to_numpy_arrays(self):
        self.data = np.asarray(self.data).astype('float32')
        self.aso_ids = np.array(self.aso_ids)

    def __create_model(self):
        x_train, x_val, y_train, y_val = train_test_split(self.data, self.aso_ids, test_size=0.2, random_state=42)

        num_classes = len(np.unique(self.aso_ids))

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(x_train.shape[1])),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))
        accuracy = model.evaluate(x_val, y_val)
        accuracy_percentage = accuracy[1] * 100
        print(f"Model accuracy: {accuracy_percentage}")

        return model

    def __decide_aso_id_equilevent_int(self, aso_id):
        match aso_id:
            case '/m/018vs':
                return 0
            case '/m/02_41':
                return 1
            case '/m/0242l':
                return 2
            case '/m/07q6cd_':
                return 3
            case '/m/03qtq':
                return 4
            case '/m/04brg2':
                return 5
            case '/m/07pbtc8':
                return 6
            case '/m/0g6b5':
                return 7
            case '/m/0bm0k':
                return 8
            case '/m/02mk9':
                return 9
            case '/m/06mb1':
                return 10
            case '/m/07rjzl8':
                return 11
            case '/m/081rb':
                return 12
            case '/m/07qcx4z':
                return 13
            case '/m/042v_gx':
                return 14
            case '/m/03m9d0z':
                return 15
            case '/m/05r5c':
                return 16
            case '/m/0l15bq':
                return 17
            case '/m/02_nn':
                return 18
            case '/m/039jq':
                return 19

    def save_model(self, model):
        joblib.dump(model, 'egitilmis_model_1.pkl')

    def train_model(self):
        self.__set_audio_files_with_labels()
        self.__load_multiple_audio_file()
        self.__convert_data_to_numpy_arrays()
        model = self.__create_model()
        self.save_model(model)

class MLAlgorithmTest():
    audio_files_with_labels = []
    trained_label_name = []
    test_csv = []

    true_counter = 0
    counter = 0

    def __init__(self):
        self.__model = joblib.load('egitilmis_model_deneme.pkl')

    def __load_audio_file(self, file_path):
        audio, sr = librosa.load("data\\test_data\\"+file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

        return np.mean(mfccs, axis=1)

    def __set_audio_files_with_labels(self):
        df = pd.read_csv('meta\\test.csv')
        self.audio_files_with_labels = df.values.tolist()

    def __load_test_data(self):
        for fname, label, aso_id in self.audio_files_with_labels:
            mfccs = self.__load_audio_file(fname)
            prediction = self.__make_prediction(mfccs)
            print(f"Predicted noise type: {fname} is {prediction}")

        print("Counter " + str(self.counter))
        print("True counter" + str(self.true_counter))

    def __get_trained_label_name(self):
        with open('meta\\aso_id_equivalent_to_label.csv', 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.trained_label_name.append(row)

    def __get_test_csv(self):
        with open('meta\\test.csv', 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.test_csv.append(row)

    def __make_prediction(self, mfccs):
        predictions = self.__model.predict(mfccs[np.newaxis, :])
        predicted_class_index = np.argmax(predictions)

        if self.test_csv[self.counter][1] == self.trained_label_name[predicted_class_index][1]:
            self.true_counter +=1
        self.counter +=1

        return self.trained_label_name[predicted_class_index][1]

    def test_data(self):
        self.__get_trained_label_name()
        self.__get_test_csv()
        self.__set_audio_files_with_labels()
        self.__load_test_data()


class MLAlgorithm():
    def __init__(self, filePath):
        self.filePath = filePath
        self.__model = joblib.load('egitilmis_model_1.pkl')
        self.meta_data = self.__set_meta_data()

    def __load_audio_file(self):
        audio, sr = librosa.load(self.filePath)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        return np.mean(mfccs, axis=1)

    def __set_meta_data(self):
        meta_data = []
        with open('meta\\aso_id_equivalent_to_label.csv', 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                meta_data.append(row)
        return meta_data

    def make_predict(self):
        mfccs = self.__load_audio_file()
        predictions = self.__model.predict(mfccs[np.newaxis, :])
        predicted_class_index = np.argmax(predictions)
        return self.meta_data[predicted_class_index][3]


# ml = MLAlgorithm("recordedfile.wav")
# predict = ml.make_predict()
# print(predict)