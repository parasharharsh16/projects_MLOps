from utills import preprocess_data,tune_hparams,read_digit,split_train_dev_test
from sklearn.model_selection import ParameterGrid
import os
from api.main import app
from sklearn import datasets
import json
import numpy as np
from joblib import load

def test_check_hpram_combination_count():
    #this test case is to check all hparam combinations are generated
    params_grid = {
    "gamma": [0.001, 0.01, 0.1, 1, 10, 100],
    "C": [0.1, 1, 2, 5, 10],
    }
    out_combinations = list(ParameterGrid(params_grid))
    assert len(out_combinations) == len(params_grid["gamma"]) * len(params_grid["C"])


def test_check_hpram_combination_values():
    #this test case is to check all hparam combinations are generated
    params_grid = {
    "gamma": [0.001, 0.01, 0.1, 1, 10, 100],
    "C": [0.1, 1, 2, 5, 10],
    }
    out_combinations = list(ParameterGrid(params_grid))
    assert len(out_combinations) == len(params_grid["gamma"]) * len(params_grid["C"])
    expectedCombo1 = {"gamma": 0.1,"C": 0.1}
    expectedCombo2 = {"gamma": 0.01,"C": 10}
    assert expectedCombo1 in out_combinations
    assert expectedCombo2 in out_combinations


def dummy_hparam():
    {
    "svm":
    {"gamma": [0.001, 0.01, 0.1, 1, 10, 100],
    "C": [0.1, 1, 2, 5, 10]},
    "dt": {
    "max_depth":[1, 2,3, 4, 5]
}
}
    params_grid = {"svm":{"gamma": [0.001, 10, 100],"C": [0.1, 5, 10],},
                    "dt": {
                    "max_depth":[1, 2,3, 4, 5]
                },
                "lr":{
                        "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
                    }
                }
    return params_grid

def test_data_splitting():
    X,y = read_digit()
    X = X[:100,:,:]
    y = y[:100]

    dev_size = 0.6
    test_size = 0.1

    train_size = 1- test_size-dev_size
    X_train,X_test,X_dev,y_train,y_test,y_dev = split_train_dev_test (X,y,test_size=0.1, dev_size=0.6)

    assert(len(X_train) ==30)
    assert(len(X_dev) ==60)
    assert(len(X_test) ==10)

def test_model_saving():
    X,y = read_digit()
    X_train = preprocess_data(X[:100,:,:])
    y_train = y[:100]
    X_dev = preprocess_data(X[:50,:,:])
    y_dev = y[:50]
    #tune_hparams(X_train, X_dev, y_train, y_dev, hyper_params)
    dummy_hparameters = dummy_hparam()
    model_path, params, dev_accu = tune_hparams(X_train,X_dev,y_train,y_dev,dummy_hparameters,model_type_name="dt")
    assert(os.path.exists(model_path)==True)

def test_get_root():
    response = app.test_client().get("/")
    assert response.status_code == 200

def test_post_predict():
    # Load the digits dataset
    digits = datasets.load_digits()
    # Dictionary to store images for each digit
    images_by_digit = {}
    # Dictionary to hold lists of images for each digit
    images_by_digit = {i: [] for i in range(10)}  # Create empty lists for digits 0-9
    # Loop through the dataset and organize images by their digit labels
    for image, label in zip(digits.images, digits.target):
        images_by_digit[label].append(image)

    #Adding assert for all the digits
    for key in images_by_digit.keys():
        processedimage = preprocess_data(np.array([(images_by_digit[key][1])]))
        imagejson = {'image': processedimage[0].tolist()}
        #response = app.test_client().post("/predict/svm",json = json.dumps(imagejson))

        #Test cases for two routes
        response = app.test_client().post("/predict/svm",json = imagejson)
        assert int(json.loads(response.data)['result']) == key
        response = app.test_client().post("/predict/lr",json = imagejson)
        assert int(json.loads(response.data)['result']) == key

#Test cases added for major exam to check if all models of LR classifier are saving or not
def test_lr_model_saving():
    model_path = "./models/M22AIE210_lr_liblinear.joblib"
    #Loading model for the testing using path that retured for best model for LR
    model = load(model_path)
    #checking the loaded model is LR model
    path_strings = model_path.split('/')
    model_name_from_path = path_strings[len(path_strings)-1]
    model_name_from_path = model_name_from_path.split(".")[0]
    model_type_from_path = model_name_from_path.split("_")[1]
    assert(model_type_from_path=="lr")
    #checking the loaded model has solver as mentioed in file name
    solver_type_from_path = model_name_from_path.split("_")[2]
    assert(solver_type_from_path== str(model.solver))