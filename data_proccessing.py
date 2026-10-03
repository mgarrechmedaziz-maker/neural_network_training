import torch 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def process():
    wine =load_iris()
    x,y= wine.data,wine.target

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y)

    scaler= StandardScaler().fit(x_train)
    scaled_x_train=scaler.transform(x_train)
    scaled_x_test=scaler.transform(x_test)

    tensor_x_train =torch.tensor(scaled_x_train,dtype=torch.float32)
    tensor_x_test =torch.tensor(scaled_x_test,dtype=torch.float32)
    tensor_y_train =torch.tensor(y_train,dtype=torch.long)
    tensor_y_test =torch.tensor(y_test,dtype=torch.long)

    return tensor_x_train ,tensor_x_test,tensor_y_train,tensor_y_test

process()