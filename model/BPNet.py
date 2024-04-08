import torch
import torch.nn as nn

class BPN(nn.Module):
    def __init__(self):
        super(BPN,self).__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(24,32, bias = True),
            nn.ReLU()
        )
        self.dense3 = nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU()
        )
        self.dense4 = nn.Sequential(
            nn.Linear(16,8),
            nn.ReLU()
        )
        self.outlayer = nn.Sequential(
            nn.Linear(8,1, bias = True),
            nn.Sigmoid()
        )
        self.initialize_weights()
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        x = self.dense1(x)
        #x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.outlayer(x)
        return x
