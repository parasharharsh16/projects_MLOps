#!/bin/bash
pytest
python code_exp.py 5 0.2 0.1 [svm,dt] config.json

# export FLASK_APP=api/main
# #flask run
# flask run --host=0.0.0.0