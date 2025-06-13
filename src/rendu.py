import os.path
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_digits
from skimage.filters import sobel
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
import keras

# The lines below shall not be modified!

# The following will be replaced by our own 
if os.path.isfile("test_data.npy"):
    X_test = np.load("test_data.npy")
    y_test = np.load("test_labels.npy")
    X_train, y_train = load_digits(return_X_y=True)
else:
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Prepare your learning pipeline and set all the parameters for your final algorithm.
class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n = X.shape[0]
        sobel_mean = np.zeros((n,1))
        for i in range(n):
            shape = int(np.sqrt(X.shape[1]))
            img = X[i].reshape(shape, shape)
            sobel_img = sobel(img)
            sobel_mean[i] = np.mean(sobel_img)
        return sobel_mean

class ZonalInfoPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n, d = X.shape
        zone_size = d // 3

        res = np.zeros((n, 3))
        for i in range(n):
            res[i, 0] = np.mean(X[i, :zone_size])
            res[i, 1] = np.mean(X[i, zone_size:2*zone_size])
            res[i, 2] = np.mean(X[i, 2*zone_size:d])
        return res

components = np.argmax(np.cumsum(PCA(X_train.shape[1]).fit(X_train).explained_variance_ratio_) >= 0.90) + 1

features = FeatureUnion([
    ('pca', PCA(n_components = components)),
    ('zones', ZonalInfoPreprocessing()),
    ('sobel', EdgeInfoPreprocessing())
])

preprocessing = Pipeline([
    ('scaler', MinMaxScaler()),
    ('features', features),
    ('postscale', StandardScaler())
])

X_train_transformed = preprocessing.fit_transform(X_train)
X_test_transformed = preprocessing.transform(X_test)

model = keras.Sequential([
        keras.layers.Input(shape=(components + 4,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),  
        keras.layers.Dense(32, activation='relu'),  
        keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# The next lines shall not be modified
model.fit(X_train_transformed, y_train, epochs=20, validation_split=0.2)
print(f"Score on the test set (loss, accuracy) {model.evaluate(X_test_transformed, y_test)}")