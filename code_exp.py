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

from utills import split_data,load_model, preprocess_data, train_module,split_train_dev_test,predict_and_eval,tune_hparams_svm,getCombinationOfParameters,read_digit,total_sample_number,size_of_image

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
X,y = read_digit()
#2. Split the dataset for train and test


X_train,X_test,X_dev,y_train,y_test,y_dev = split_train_dev_test (X,y,test_size=0.2, dev_size=0.1)
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)

# 3. Preprocessing the data
X_test = preprocess_data(X_test)
X_train = preprocess_data(X_train)

X_dev = preprocess_data(X_dev)

# 4. Model training
# Create a classifier: a support vector classifier
# Learn the digits on the train subset


params_grid = {
    "gamma": [0.001, 0.01, 0.1, 1, 10, 100],
    "C": [0.1, 1, 2, 5, 10],
}
test_size =[0.1, 0.2, 0.3]
dev_size = [0.1, 0.2, 0.3]


for test,dev in getCombinationOfParameters(test_size,dev_size):
    train = 1-(test+dev)
    X_train,X_test,X_dev,y_train,y_test,y_dev = split_train_dev_test (X,y,test_size=test, dev_size=dev)
    X_test = preprocess_data(X_test)
    X_train = preprocess_data(X_train)
    X_dev = preprocess_data(X_dev)

    model_path, params, dev_accu = tune_hparams_svm(X_train,X_dev,y_train,y_dev,params_grid)
    model = load_model(best_model_path= model_path)
    
    train_accu = predict_and_eval(model,X_train,y_train)
    test_accu = predict_and_eval(model,X_test,y_test)
    #Commented print for Quiz so it can print current output
    #print( f"test_size={test} dev_size={dev} train_size={train} train_acc={train_accu} dev_acc={dev_accu} test_acc={test_accu}" )

#Code for the Quiz
#print(f"Total samples in datasets are {total_sample_number(X)}")

#height, width = size_of_image(X)

#print(f"Height of input image is {height} and width of image is {width}")
