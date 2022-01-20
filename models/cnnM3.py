import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnM3(nn.Module):
    def __init__(self):
        super(CnnM3, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)      
        self.conv_layer1_bn = nn.BatchNorm2d(num_features=32)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3) 
        self.conv_layer2_bn = nn.BatchNorm2d(num_features=48)
        self.conv_layer3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3) 
        self.conv_layer3_bn = nn.BatchNorm2d(num_features=64)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3) 
        self.conv_layer4_bn = nn.BatchNorm2d(num_features=80)
        self.conv_layer5 = nn.Conv2d(in_channels=80, out_channels=96, kernel_size=3) 
        self.conv_layer5_bn = nn.BatchNorm2d(num_features=96)
        self.conv_layer6 = nn.Conv2d(in_channels=96, out_channels=112, kernel_size=3)  
        self.conv_layer6_bn = nn.BatchNorm2d(num_features=112)
        self.conv_layer7 = nn.Conv2d(in_channels=112, out_channels=128, kernel_size=3) 
        self.conv_layer7_bn = nn.BatchNorm2d(num_features=128)
        self.conv_layer8 = nn.Conv2d(in_channels=128, out_channels=144, kernel_size=3) 
        self.conv_layer8_bn = nn.BatchNorm2d(num_features=144)
        self.conv_layer9 = nn.Conv2d(in_channels=144, out_channels=160, kernel_size=3)  
        self.conv_layer9_bn = nn.BatchNorm2d(num_features=160)
        self.conv_layer10 = nn.Conv2d(in_channels=160, out_channels=176, kernel_size=3)
        self.conv_layer10_bn = nn.BatchNorm2d(num_features=176)
        self.linear_layer1 = nn.Linear(in_features=11264, out_features=10)
        self.linear_layer1_bn = nn.BatchNorm1d(num_features=10)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer1_bn(x)
        x = F.relu(x)
        
        x = self.conv_layer2(x)
        x = self.conv_layer2_bn(x)
        x = F.relu(x)
        
        x = self.conv_layer3(x)
        x = self.conv_layer3_bn(x)
        x = F.relu(x)
        
        x = self.conv_layer4(x)
        x = self.conv_layer4_bn(x)
        x = F.relu(x)


        x = self.conv_layer5(x)
        x = self.conv_layer5_bn(x)
        x = F.relu(x)

        x = self.conv_layer6(x)
        x = self.conv_layer6_bn(x)
        x = F.relu(x)

        x = self.conv_layer7(x)
        x = self.conv_layer7_bn(x)
        x = F.relu(x)


        x = self.conv_layer8(x)
        x = self.conv_layer8_bn(x)
        x = F.relu(x)        
        
        x = self.conv_layer9(x)
        x = self.conv_layer9_bn(x)
        x = F.relu(x)

        x = self.conv_layer10(x)
        x = self.conv_layer10_bn(x)
        x = F.relu(x)

        x = torch.flatten(x,start_dim=1)
        #print(" > x :",x.size())

        x = self.linear_layer1(x)
        x = self.linear_layer1_bn(x)

        output = F.log_softmax(x, dim=1)

        return output
