from utills import tune_hparams,read_digit,split_train_dev_test
from sklearn.model_selection import ParameterGrid
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


