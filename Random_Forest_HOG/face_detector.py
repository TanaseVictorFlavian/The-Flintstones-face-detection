from skimage.feature import hog
import cv2 as cv
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import time


class DataSet:
    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


class FacialDetector:
    def __init__(self, positive_path, negative_path) -> None:
        self.positive_images_path = positive_path
        self.negative_images_path = negative_path
        self.best_model = None

    def get_descriptors(self, images_path: str, resize_shape: tuple = (128, 128), hog_cell_dim: int = 8):
        # this function's purpose is to calculate the HOG descriptors of the negative and positive images

        files = [images_path + x for x in (os.listdir(images_path))]
        descriptors = []  # -> list of numpy arrays

        for file in files:
            # read the images as grayscale
            img = cv.imread(file, cv.IMREAD_GRAYSCALE)

            # resize to the desired shape
            img = cv.resize(img, resize_shape)

            # calculate hog
            features = hog(img, pixels_per_cell=(
                hog_cell_dim, hog_cell_dim), cells_per_block=(2, 2), feature_vector=True)
            descriptors.append(features)

        return np.array(descriptors)

    def train_Random_Forest(self):
        print("%% Extracting features ... %%")
        positive_descriptors = self.get_descriptors(self.positive_images_path)
        negative_descriptors = self.get_descriptors(self.negative_images_path)

        DataSet = self.create_data_set(
            positive_descriptors, negative_descriptors)

        X_train, y_train, X_test, y_test = DataSet.X_train, DataSet.y_train, DataSet.X_test, DataSet.y_test

        # param_distribution for random_search

        param_dist = {
            'n_estimators': [int(x) for x in np.linspace(start=100, stop=500, num=20)],
            'max_depth': [None] + [int(x) for x in np.linspace(5, 50, num=10)],
            'min_samples_split': [int(x) for x in np.linspace(2, 20, num=10)],
            'min_samples_leaf': [int(x) for x in np.linspace(1, 20, num=10)],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
        }

        model = RandomForestClassifier()

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=10,
            cv=5,
            scoring='accuracy',
            random_state=123
        )

        print("% Searching for best parameters %")

        start_time = time.time()
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        end_time = time.time()

        joblib.dump(best_model, 'random_forest_facial_detector_HOG.joblib')
        self.best_model = best_model

        print(
            f"% Randomized search time elapsed : {end_time - start_time} seconds %")
        print("Best parameters:\n")
        print(random_search.best_params_)
        print()

        self.evaluate_model(best_model, X_test, y_test)

    def create_data_set(self, positive_features, negative_features):
        # label with 1 positive features and 0 negative features
        positive_labels = np.ones(len(positive_features))
        negative_labels = np.zeros(len(negative_features))

        features = np.concatenate([positive_features, negative_features])
        labels = np.concatenate([positive_labels, negative_labels])

        # shuffle the data and split it into train and validation sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=123, shuffle=True)

        return DataSet(X_train, y_train, X_test, y_test)

    def evaluate_model(self, model, X_test, y_true):
        # test model performance
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy}")

        precision = precision_score(y_true, y_pred)
        print(f"Precision: {precision}")

        recall = recall_score(y_true, y_pred)
        print(f"Recall: {recall}")

        f1 = f1_score(y_true, y_pred)
        print(f"F1 Score: {f1}\n")


if __name__ == "__main__":

    facial_detector = FacialDetector(
        "./Positive Images/", "./Negative Images/")
    facial_detector.train_Random_Forest()
