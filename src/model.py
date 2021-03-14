import torch
import torch.nn as nn
import config

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, (5,5), 1)
        self.c2 = nn.Conv2d(32, 32, (5,5), 1)
        self.c3 = nn.Conv2d(32, 64, (3,3), 1)
        self.c4 = nn.Conv2d(64, 64, (3,3), 1)
        self.fc1 = nn.Linear(21504, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d((2,2))
        self.max_pool2 = nn.MaxPool2d((2,2), (2,2))
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.relu(self.c2(x))
        x = self.dropout1(self.max_pool1(x))
        x = self.relu(self.c3(x))
        x = self.relu(self.c4(x))
        x = self.dropout1(self.max_pool2(x))
        x = self.dropout2(self.relu(self.fc1(x.view(x.size(0), -1))))
        x = self.fc2(x)
        
        return x

device = torch.device("cpu")

def load_model():
    model = Net()
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location={'cuda:0': 'cpu'}))
    model.to(device)
    model.eval()
    return model