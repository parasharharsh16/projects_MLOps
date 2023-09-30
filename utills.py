from sklearn.model_selection import train_test_split
from sklearn import svm,metrics
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import ParameterGrid
import itertools
from sklearn import datasets, metrics
#Util defination
# flatten the images
def read_digit():
    digits = datasets.load_digits()
    data = digits.images
    return data,digits.target


def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples,-1))
    return data

def split_data(x,y,test_size,random_state=1):
    X_train,X_test,y_train,y_test = train_test_split(
        x,y,test_size=test_size,random_state=random_state
    )
    return X_train,X_test,y_train,y_test

def train_module(x,y,model_params,model_type = "svm"):
    if model_type == "svm":
        #create a classifier: a support vector classifier
        clf = svm.SVC
    model = clf(**model_params)
    #train the model
    model.fit(x,y)
   # if x_dev.any()!= None and y_dev.any() != None:
    # accu = model.score(x_dev,y_dev)
    # print("Accuracy on validiation data of this model is "+str(round(accuracy,3)*100)+"%\n\n")
    return model

def split_train_dev_test(x, y, test_size, dev_size,random_state=1):
    #spliting for train and test
    
    # X_train_temp,X_test,y_train_temp,y_test = train_test_split(
    #     x,y,test_size=test_size,random_state=random_state
    # )

    X_train_temp,X_test,y_train_temp,y_test = split_data(x,y,test_size,random_state)
    
    # Inorder to overcome the fraction issue with dual splitting, we are getiing correct proportion for dev set after alreading spliting test set
    dev_size = ((len(x)*dev_size)/len(X_train_temp))

    X_train,X_dev,y_train,y_dev = split_data(X_train_temp,y_train_temp,dev_size,random_state)

    return X_train,X_test,X_dev,y_train,y_test,y_dev

def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    # print(
    # f"Classification report for classifier {model}:\n"
    # f"{metrics.classification_report(y_test, predicted)}\n"
    # )

    # disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    # disp.figure_.suptitle("Confusion Matrix")
    #print(f"Confusion matrix:\n{disp.confusion_matrix}")

    # plt.show()


    # # The ground truth and predicted lists
    # y_true = []
    # y_pred = []
    # cm = disp.confusion_matrix

    # # For each cell in the confusion matrix, add the corresponding ground truths
    # # and predictions to the lists
    # for gt in range(len(cm)):
    #     for pred in range(len(cm)):
    #         y_true += [gt] * cm[gt][pred]
    #         y_pred += [pred] * cm[gt][pred]

    # print(
    #     "Classification report rebuilt from confusion matrix:\n"
    #     f"{metrics.classification_report(y_true, y_pred)}\n"
    # )


    return metrics.accuracy_score(y_test,predicted)

def tune_hparams(X_train, X_dev, y_train, y_dev, hyper_params):
    best_accu = -1
    optimized_model = None
    best_params = {}
    for hyper_params in ParameterGrid(hyper_params):
        current_model = train_module(X_train, y_train, hyper_params, model_type="svm")
        current_accu = predict_and_eval(current_model, X_dev, y_dev)

        if current_accu > best_accu:
            best_accu = current_accu
            best_params = hyper_params
            optimized_model = current_model
    return optimized_model,best_params,best_accu

def getCombinationOfParameters(test_size,dev_size):
    return list(itertools.product(test_size,dev_size))

