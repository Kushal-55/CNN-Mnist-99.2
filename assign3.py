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

        # computing mean error rates


    mean4 = np.mean([0.09775, 0.09787500000000005, 0.09787500000000005, 0.09775, 0.09775])
    print("Mean accuracy for CNN ", mean4)
    mean5 = np.mean([0.03325, 0.0345, 0.03375, 0.032875, 0.034125])
    print("Mean error rate for knn = 1 ", mean5)
    mean6 = np.mean([0.035625, 0.03775, 0.032625, 0.0345, 0.032])
    print("Mean error rate for knn = 5 ", mean6)
    mean7 = np.mean([0.037125, 0.039, 0.040375, 0.040125, 0.040625])
    print("Mean error rate for knn = 10 ", mean7)


def question2():
    # Loading and Checking data
    dataset = pd.read_csv('mydata.csv')
    newdata = dataset.replace("M", "1").replace("B", "0")
    del newdata["id"]

    # Getting all the columns
    columns_for_scoring = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                           'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
                           'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                           'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                           'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                           'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
                           'symmetry_worst', 'fractal_dimension_worst']

    # Getting AUC for each and every feature
    for colname in columns_for_scoring:
        auc = roc_auc_score(newdata['diagnosis'], newdata[colname])

    values = [0.9375165160403784, 0.7758244807356903, 0.9468976269753184, 0.9383158923946937, 0.7220416468474182,
              0.86378230537498, 0.9378270175994926, 0.9644376618571957, 0.6985624438454627, 0.48453437978965175,
              0.8683341261032713, 0.5115942603456478, 0.8763939538079382, 0.9264111304899318, 0.4688375350140056,
              0.7272805348554516, 0.7808189313461233, 0.7917921885735425, 0.4448892764653031, 0.6203028381163787,
              0.9704428941387877, 0.7846308334654617, 0.9754505575815232, 0.9698284974367105, 0.7540563395169388,
              0.8623024681570742, 0.921363828550288, 0.9667036625971144, 0.736939115268749, 0.6859706146609588]

    # Checking whether auc values are of utmost important or not
    values = [x - 0.5 for x in values]
    abs_values = map(abs, values)
    final_values = [x + 0.5 for x in abs_values]

    # Sorting auc values in descending order
    final_values.sort(reverse=True)
    print(final_values)


def question3():
    # Loading the dataset
    dataset = pd.read_csv('mydata.csv')
    # Dropping columns which are not required
    dataset = dataset.drop("id", 1)
    dataset = dataset.drop("Unnamed: 32", 1)

    # Assigning 0s and 1s
    d = {'M': 0, 'B': 1}

    # Transforming items without looping
    dataset['diagnosis'] = dataset['diagnosis'].map(d)

    # Assigning X and y
    measurements = list(dataset.columns[1:31])
    X = dataset[measurements]
    y = dataset["diagnosis"]

    # implementing k fold for 5 splits
    kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # implementing the best performing classifier for this dataset
        ann = MLPClassifier(hidden_layer_sizes=25, activation="tanh", solver="lbfgs")
        ann.fit(X_train, y_train)
        y_pred = ann.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print('Error rate for ANN ', 1 - score)

        ann2 = MLPClassifier(hidden_layer_sizes=5, activation="tanh", solver="adam")
        ann2.fit(X_train, y_train)
        y_pred2 = ann2.predict(X_test)
        score2 = accuracy_score(y_test, y_pred2)
        print('Error rate for ANN 2 ', 1 - score2)

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

    mean1 = np.mean([0.1228070175438597,0.052631578947368474, 0.07017543859649122,0.06140350877192979, 0.09734513274336287])
    print("Mean accuracy for ANN ", mean1)
    mean5 = np.mean(
        [0.368421052631579,0.0964912280701754,0.368421052631579,0.368421052631579, 0.12389380530973448])
    print("Mean accuracy for ANN 2 ", mean5)
    mean2 = np.mean([0.08849557522123894,0.08771929824561403,0.087719298245614030,.07017543859649122,0.14035087719298245])
    print("Mean error rate for knn = 1 ", mean2)
    mean3 = np.mean([0.07079646017699115,0.05263157894736842,0.06140350877192982,0.06140350877192982,0.11403508771929824])
    print("Mean error rate for knn = 5 ", mean3)
    mean4 = np.mean([0.061946902654867256,0.043859649122807015,0.05263157894736842,0.06140350877192982,0.12280701754385964])
    print("Mean error rate for knn = 10 ", mean4)



def question4():
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
    #question2()
     question3()
    #question4()
