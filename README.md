# how to setup the environment
install conda 
conda create -n lecture3 python==3.9
conda activate lecture3
pip install -r requriremnts.txt

# How to run
CLI will have params max_run(int), dev_size(float),test_size(float), model_type(list)-either dt or svm, config path for params <br/>
- python code_exp.py max_run,dev_size,test_size,model_type, config_path <br/>
- e.g. python code_exp.py 5 0.2 0.1 [svm,dt,lr] config.json <br/>

# meaning of failure
- Poor performance in validation
- compile time/runtime error in code
- Additional comment for branch change
