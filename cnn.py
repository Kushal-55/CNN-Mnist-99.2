# importing libraries

from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import svm
from sklearn.model_selection import train_test_split
import scipy.io
from sklearn.neural_network import MLPClassifier
from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization
from tensorflow.keras.layers import Dense, ReLU
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from pathlib import Path


def question1():
    # loading dataset
    dataset = loadmat(r"C:\Users\Kushal\Downloads\NumberRecognitionBiggest.mat")

    # xtrain contains images, ytrain contains labels
    images = dataset["X_train"]
    labels = dataset["y_train"].squeeze()

    # using stratified k fold for 5 splits
    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)  # split as training and testing sets

    for train_index, test_index in skf.split(images, labels):
        X_train, X_test = images[train_index], images[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # reshaping for cnn
        Xtrain = X_train.reshape(X_train.shape[0], 28, 28)
        Xtest = X_test.reshape(X_test.shape[0], 28, 28)
        Xtrain = np.expand_dims(Xtrain, axis=-1)
        Xtest = np.expand_dims(Xtest, axis=-1)

        # changing datatype to float for cnn
        Xtrain = Xtrain.astype('float32') / 255.0
        Xtest = Xtest.astype('float32') / 255.0

        # implementing cnn
        model = Sequential()
        model.add(BatchNormalization())
        model.add(
            Conv2D(5, kernel_size=3, input_shape=(28, 28, 1),
                   activation="linear",
                   data_format="channels_last")
        )
        model.add(ReLU())
        model.add(Flatten())
        model.add(
            Dense(units=10, activation="softmax")
        )
        model.compile(
            optimizer=optimizers.SGD(momentum=0.9, lr=0.001),
            loss=losses.mean_squared_error,
            metrics=["accuracy"],
        )
        # fitting the model
        fitting = model.fit(Xtrain.astype(float), y_train, epochs=15, verbose=1)

        # evaluating the model
        ypred1 = model.evaluate(X_test)

        # argmax returns max value of an array
        predictions = np.argmax(ypred1, axis=-1)

        # computing accuracy
        cnnerror = (1 - np.mean(y_test != predictions))
        print("Accuracy for CNN:", cnnerror)

        # reshaping into 2D for KNN
        X_train = X_train.reshape((X_train.shape[0], 784))
        X_test = X_test.reshape((X_test.shape[0], 784))

        # implementing knn = 1, 5 ,10
        knn_1 = KNeighborsClassifier(n_neighbors=1)
        knn_1.fit(X_train, y_train)
        y_pred2 = knn_1.predict(X_test)
        knn1error = np.mean(y_pred2 != y_test)  # mean error
        print('Error rate for knn 1 =', knn1error)

        knn_5 = KNeighborsClassifier(n_neighbors=5)
        knn_5.fit(X_train, y_train)
        y_pred3 = knn_5.predict(X_test)
        knn5error = np.mean(y_pred3 != y_test)  # mean error
        print('Error rate for knn 5 =', knn5error)

        knn_10 = KNeighborsClassifier(n_neighbors=10)
        knn_10.fit(X_train, y_train)
        y_pred4 = knn_10.predict(X_test)
        knn10error = np.mean(y_pred4 != y_test)  # mean error
        print('Error rate for knn 10 =', knn10error)

        # compute mean error rates

# second attempt on maximizing cnn accuracy
def question2():
    # loading dataset
    dataset = loadmat(r"C:\Users\Kushal\Downloads\NumberRecognitionBiggest.mat")
    images1 = dataset['X_train']
    labels = dataset['y_train']
    images2 = dataset['X_test']

    # reshaping for cnn
    Xtrain = images1.reshape((images1.shape[0], images1.shape[1], images1.shape[2], 1))
    Xtest = images2.reshape((images2.shape[0], images2.shape[1], images2.shape[2], 1))
    ytrain = labels.reshape((labels.shape[1],))

    # changing datatype to float for cnn
    Xtrain = Xtrain.astype('float32') / 255.0
    Xtest = Xtest.astype('float32') / 255.0
    img_shape = Xtrain.shape[1:]

    # applying stratified k fold
    kf = StratifiedKFold(n_splits=5, random_state=None)

    for train_index, test_index in kf.split(Xtrain, ytrain):
        X_train, X_test = Xtrain[train_index], Xtrain[test_index]
        y_train, y_test = ytrain[train_index], ytrain[test_index]

        # defining input shape
        inputshape = X_train.shape[1:]

        # implementing model
        model = Sequential()
        # adding conv2D layer
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=inputshape))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(48, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        # addding dropout layer
        model.add(Dropout(0.5))

        # flattening the data
        model.add(Flatten())

        # adding final layer/dense layer
        model.add(Dense(500, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        # using optimizer as 'adam' for optimum performance
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # high epochs and batch size for better predictions
        fitting = model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=2, validation_split=0.1)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test Accuracy:{accuracy * 100}')
        print(f'Error Rate:{1- accuracy}')
        # predicting
        y_pred = model.predict([Xtest])
        print('Predicted: {}'.format((y_pred)))
        y_pred = y_pred*255
        y_pred = np.argmax(y_pred, axis = -1)
        y_pred = np.uint8(y_pred)


if __name__ == "__main__":
    #question1()
    question2()
 
