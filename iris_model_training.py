import torch 
import torch.nn as nn
import torch.optim as optim 
from data_proccessing import process
from sklearn.metrics import accuracy_score

x_train,x_test,y_train,y_test = process()

model=nn.Sequential(
    nn.Linear(in_features=4,out_features=20),
    nn.ReLU(),
    nn.Linear(in_features=20,out_features=5),
    nn.ReLU(),
    nn.Linear(in_features=5,out_features=3)
)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

num_epochs=10000
for epoch in range(1,num_epochs+1):
    model.train()
    optimizer.zero_grad()
    outputs=model(x_train)
    loss=criterion(outputs,y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0 : 
        with torch.no_grad() :
            model.eval()
            outputs=model(x_test)
            loss_val=criterion(outputs,y_test)
            _,prediction= torch.max(outputs,1)
            accuracy=accuracy_score(prediction,y_test.numpy())
            print("in epoch: ",epoch," the loss= ",loss_val.item()," and the accuracy= ",accuracy)


