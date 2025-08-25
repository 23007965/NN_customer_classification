# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:

### STEP 3:


## PROGRAM

### Name: 
### Register Number:

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  # First hidden layer
        self.fc2 = nn.Linear(16, 8) # Second hidden layer
        self.fc3 = nn.Linear(8, 4) # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


```
```python

Parthiban = PeopleClassifier(input_size=x_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Parthiban.parameters(),lr=0.01)
```
```python

def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
      model.train()
      for X_batch,y_batch in train_loader:
        optimizer.zero_grad()
        outputs=model(X_batch)
        loss=criterion(outputs,y_batch)
        loss.backward()
        optimizer.step()


    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    #Include your code here
```



## Dataset Information

Include screenshot of the dataset

## OUTPUT



### Confusion Matrix

<img width="816" height="568" alt="image" src="https://github.com/user-attachments/assets/5602d3eb-078f-4d2c-ae06-14399e690904" />


### Classification Report

<img width="598" height="249" alt="image" src="https://github.com/user-attachments/assets/5762ea38-09bd-409a-b675-082e7f94aec3" />



### New Sample Data Prediction

Include your sample input and output here

## RESULT
Include your result here
