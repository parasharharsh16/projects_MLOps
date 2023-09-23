from Utills import tune_hparams
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
