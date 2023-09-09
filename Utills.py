from sklearn.model_selection import train_test_split
from sklearn import svm,metrics
import matplotlib.pyplot as plt
from sklearn import svm
#Util defination
# flatten the images
def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples,-1))
    return data

def split_data(x,y,test_size,random_state=1):
    X_train,X_test,y_train,y_test = train_test_split(
        x,y,test_size=test_size,random_state=random_state
    )
    return X_train,X_test,y_train,y_test

def train_module(x,y,model_params, x_dev =None, y_dev =None, model_type = "sva"):
    if model_type == "svm":
        #create a classifier: a support vector classifier
        clf = svm.SVC
    model = clf(**model_params)
    #train the model
    model.fit(x,y)
   # if x_dev.any()!= None and y_dev.any() != None:
    # accuracy = model.score(x_dev,y_dev)
    # print("Accuracy on validiation data of this model is "+str(round(accuracy,3)*100)+"%\n\n")
    return model

def split_train_dev_test(x, y, test_size, dev_size,random_state=1):
    #spliting for train and test
    
    # X_train_temp,X_test,y_train_temp,y_test = train_test_split(
    #     x,y,test_size=test_size,random_state=random_state
    # )

    X_train_temp,X_test,y_train_temp,y_test = split_data(x,y,test_size,random_state)
    
    
    # #spliting for train and dev form available train data
    
    # X_train,X_dev,y_train,y_dev = train_test_split(
    #     X_train_temp,y_train_temp,test_size==dev_size,random_state=random_state
    # )
    X_train,X_dev,y_train,y_dev = split_data(X_train_temp,y_train_temp,dev_size,random_state)

    return X_train,X_test,X_dev,y_train,y_test,y_dev

def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()


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

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )


    return model
