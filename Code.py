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
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)
from Utills import split_data, preprocess_data, train_module

#2. Split the dataset for train and test
data = digits.images
X_train, X_test, y_train, y_test = split_data(data,digits.target,test_size=0.3)

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


model = train_module(X_train,y_train,{'gamma': 0.001},model_type="svm")



# Predict the value of the digit on the test subset
predicted = model.predict(X_test)

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

# 5. Sanity check

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:


# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]


