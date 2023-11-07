#!/bin/bash
pytest
#python code_exp.py 5 0.2 0.1 [svm,dt] config.json
#Export FLASK_APP = main.py
export FLASK_APP=api/main
#flask run
flask run --host=0.0.0.0