from sklearn.model_selection import train_test_split
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

def train_module(x,y,model_params, model_type = "sva"):
    if model_type == "svm":
        #create a classifier: a support vector classifier
        clf = svm.SVC
    model = clf(**model_params)
    #train the model
    model.fit(x,y)
    return model
