from utills import preprocess_data,tune_hparams,read_digit,split_train_dev_test
from sklearn.model_selection import ParameterGrid
import os
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
                }}
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

    




