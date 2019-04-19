import time
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Definining the training routine
def train_model(model,criterion,num_epochs,learning_rate, trainLoader, use_gpu):
        start = time.time()
        train_loss = [] #List for saving the loss per epoch     
        
        for epoch in range(num_epochs):
            epochStartTime = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            running_loss = 0.0           
            # Loading data in batches
            batch = 0
            for data in trainLoader:
                inputs,labels = data
                # Wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.float().cuda()), \
                        Variable(labels.float().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)    
                # Initializing model gradients to zero
                model.zero_grad() 
                # Data feed-forward through the network
                outputs = model(inputs)
                # Predicted class is the one with maximum probability
                _, preds = torch.max(outputs.data, 1)
                # Finding the MSE
                loss = criterion(outputs, labels)
                # Accumulating the loss for each batch
                running_loss += loss.data[0]
                # Backpropaging the error
                if batch == 0:
                    totalLoss = loss
                    totalPreds = preds
                    batch += 1                    
                else:
                    totalLoss += loss
                    totalPreds = torch.cat((totalPreds,preds),0)  
                    batch += 1
                    
            totalLoss = totalLoss/batch
            totalLoss.backward()
            
            # Updating the model parameters
            for f in model.parameters():
                f.data.sub_(f.grad.data * learning_rate)                
           
            epoch_loss = running_loss/37000  #Total loss for one epoch
            train_loss.append(epoch_loss) #Saving the loss over epochs for plotting the graph           
            
            print('Epoch loss: {:.6f}'.format(epoch_loss))
            epochTimeEnd = time.time()-epochStartTime
            print('Epoch complete in {:.0f}m {:.0f}s'.format(
            epochTimeEnd // 60, epochTimeEnd % 60))
            print('-' * 25)
            # Plotting Loss vs Epochs
            fig1 = plt.figure(1)        
            plt.plot(range(epoch+1),train_loss,'r--',label='train')        
            if epoch==0:
                plt.legend(loc='upper left')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
            fig1.savefig('ConvNet_lossPlot.png')             

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        return model