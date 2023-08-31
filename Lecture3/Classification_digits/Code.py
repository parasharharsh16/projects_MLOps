"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics
from Utills import split_data, preprocess_data, train_module,split_train_dev_test,predict_and_eval

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.


# 1. Load data
digits = datasets.load_digits()

#2. Split the dataset for train and test
data = digits.images
#X_train, X_test, y_train, y_test = split_data(data,digits.target,test_size=0.3)

X_train,X_test,X_dev,y_train,y_test,y_dev = split_train_dev_test (data,digits.target,test_size=0.2, dev_size=0.1)

# 3. Preprocessing the data
X_test = preprocess_data(X_test)
X_train = preprocess_data(X_train)
X_dev = preprocess_data(X_dev)

# 4. Model training
# Create a classifier: a support vector classifier
# Learn the digits on the train subset

model = train_module(X_train,y_train,{'gamma': 0.001},X_dev,y_dev,model_type="svm")

# Predict the value of the digit on the test subset
predict_and_eval(model,X_test,y_test)

