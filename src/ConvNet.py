import torch
#from torch.autograd import Variable
import torch.nn.functional as F

class ConvNetwork(torch.nn.Module):
    
    #Our batch shape for input x is (1, 28, 28)
    
    def __init__(self):
        super(ConvNetwork, self).__init__()
        
        #Input channels = 1, output channels = 16
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
        
        #Input channels = 16, output channels = 32,  Input = (16, 14, 14)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)

        #Input channels = 32, output channels = 32,  Input = (32, 7, 7)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=0)

        
        #288 input features, 10 classes 
        self.fc = torch.nn.Linear(32 * 3 * 3, 10)

        #self.softmax = torch.nn.Softmax()
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (1, 28, 28) to (16, 28, 28)
        x = F.relu(self.conv1(x))
        
        #Size changes from (16, 28, 28) to (16, 14, 14)
        x = self.pool1(x)
        
        #Second Convolution layer , Size changes from (16, 14, 14) to (32, 14, 14)
        x = F.relu(self.conv2(x))
        
        #Size changes from (32, 14, 14) to (32, 7, 7)
        x = self.pool2(x)

        #Third Convolution layer , Size changes from (32, 7, 7) to (32, 7, 7)
        x = F.relu(self.conv3(x))
        
        #Size changes from (32, 7, 7) to (32, 3, 3)
        x = self.pool3(x)

        #Reshape data to input to the input layer of the neural net
        #Size changes from (32, 3, 3) to (1, 288)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 32 * 3 *3)
        
        #Size changes from (1, 288) to (1, 10)
        x = F.softmax(self.fc(x))
        
        return(x)