
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics

from utills import split_data,load_model, preprocess_data, train_module,split_train_dev_test,predict_and_eval,tune_hparams,getCombinationOfParameters,read_digit,total_sample_number,size_of_image

import pandas as pd
import sys 
import json
from sklearn.metrics import confusion_matrix
import numpy as np

# python code_exp.py ~max_run, dev_size, test_size, model_type
#names  args
#package name args parse
max_run = int(sys.argv[1])
dev_size = float(sys.argv[2])  
test_size = float(sys.argv[3])
n = len(sys.argv[4]) 
model_types = sys.argv[4][1:n-1] 
model_types = model_types.split(',')
config_file_Path = sys.argv[5]


X,y = read_digit()
#2. Split the dataset for train and test


X_train,X_test,X_dev,y_train,y_test,y_dev = split_train_dev_test (X,y,test_size=0.2, dev_size=0.1)


# 3. Preprocessing the data
X_test = preprocess_data(X_test)
X_train = preprocess_data(X_train)

X_dev = preprocess_data(X_dev)


with open(config_file_Path) as json_file:
    params_grid = json.load(json_file)

results_disc = []
model_disc = {}
model_accu  = {}
model_confusion = {}


for i in range(max_run):
    for model_name in model_types:
        for test,dev in getCombinationOfParameters([test_size],[dev_size]):
            train = 1-(test+dev)
            X_train,X_test,X_dev,y_train,y_test,y_dev = split_train_dev_test (X,y,test_size=test, dev_size=dev)
            X_test = preprocess_data(X_test)
            X_train = preprocess_data(X_train)
            X_dev = preprocess_data(X_dev)

            model_path, params, dev_accu = tune_hparams(X_train,X_dev,y_train,y_dev,params_grid,model_type_name=model_name)
            model = load_model(best_model_path= model_path)

            train_accu = predict_and_eval(model,X_train,y_train)
            test_accu = predict_and_eval(model,X_test,y_test)
            current_result = {"current run" : i,"model type": model_name, "test_size": test ,"dev_size": dev, "train_size":train, "train_acc":train_accu, "dev_acc":dev_accu, "test_acc":test_accu}
            results_disc.append(current_result)
            model_disc [model_name] = model
            model_accu [model_name] = f"train_acc :{train_accu} and test_acc: {test_accu}"
            
            



print("Lets consider SVM model as production model and DT model as new candidate\n")

for model_name in model_types:
    print(f"Model accuracy for {model_name} is\n {model_accu[model_name]}")

print("Confusion matrix considering SVM model result as true labels")

y_true = (model_disc["svm"]).predict(X_test)
y_pred_candidate = (model_disc["dt"]).predict(X_test)
model_confusion_matrix = confusion_matrix(y_true, y_pred_candidate, labels=np.arange(10))

print("Confusion matrix is given below: \n")
print(model_confusion_matrix)

