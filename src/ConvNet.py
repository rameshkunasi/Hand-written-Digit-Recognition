import torch
#from torch.autograd import Variable
import torch.nn.functional as F

class ConvNetwork(torch.nn.Module):
    
    #Our batch shape for input x is (1, 28, 28)
    
    def __init__(self):
        super(ConvNetwork, self).__init__()
        
        #Input channels = 1, output channels = 16
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
        
        #Input channels = 16, output channels = 32,  Input = (16, 14, 14)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)

        #Input channels = 32, output channels = 32,  Input = (32, 7, 7)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=0)

        
        #288 input features, 10 classes 
        self.fc1 = torch.nn.Linear(128 * 3 * 3, 256)
        self.fc2 = torch.nn.Linear(256, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        self.drop = torch.nn.Dropout2d(p=0.1)

        #self.softmax = torch.nn.Softmax()
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (1, 28, 28) to (16, 28, 28)
        x = F.relu(self.drop(self.bn1(self.conv1(x))))
         
        #Size changes from (16, 28, 28) to (16, 14, 14)
        x = self.pool1(x)
        
        #Second Convolution layer , Size changes from (16, 14, 14) to (32, 14, 14)
        x = F.relu(self.drop(self.bn2(self.conv2(x))))
        
        #Size changes from (32, 14, 14) to (32, 7, 7)
        x = self.pool2(x)

        #Third Convolution layer , Size changes from (32, 7, 7) to (32, 7, 7)
        x = F.relu(self.drop(self.bn3(self.conv3(x))))
        
        #Size changes from (32, 7, 7) to (32, 3, 3)
        x = self.pool3(x)

        #Reshape data to input to the input layer of the neural net
        #Size changes from (32, 3, 3) to (1, 288)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 128 * 3 *3)
        
        #Size changes from (1, 288) to (1, 10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        
        return(x)
