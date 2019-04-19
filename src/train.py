import time
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Definining the training routine
def train_model(model, optimizer, criterion,num_epochs,learning_rate, trainLoader, validLoader, use_gpu):
        
        #start = time.time()
        train_loss = [] #List for saving the loss per epoch     
        val_loss = []
        n_batches = len(trainLoader)
        
        training_start_time = time.time()

        for epoch in range(num_epochs):
            epochStartTime = time.time()
            running_loss = 0.0
            total_train_loss = 0
            print_every = n_batches // 10
            start_time = time.time()
            #batch = 0
            for i, data in enumerate(trainLoader, 0):
                inputs,labels = data
                # Wrap them in Variable
                if use_gpu:
                    #print("Converting input into float cuda\n")
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)    
                # Initializing model gradients to zero
                #print("input shape:: ",inputs.shape)
                optimizer.zero_grad() 
                # Data feed-forward through the network
                outputs = model(inputs)
                #print("output s:: ",outputs)
                # Predicted class is the one with maximum probability
                _, preds = torch.max(outputs.data, 1)
                # Finding the MSE
                loss = criterion(outputs, labels)
                loss.backward()
                # Update the network parameters
                optimizer.step()
                # Accumulate loss per batch
                #running_loss += loss.data[0]
                running_loss += loss.item()
                total_train_loss += loss.item()
                
                if (i + 1) % (print_every + 1) == 0:
                    #print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        #epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, 
                        #time.time() - start_time))
                    #Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()
           
            epoch_loss = total_train_loss/n_batches  #Total loss for one epoch
            train_loss.append(epoch_loss) #Saving the loss over epochs for plotting the graph           
            
            print('Epoch : {:.0f} Epoch loss: {:.6f}'.format(epoch, epoch_loss))
            epochTimeEnd = time.time()-epochStartTime
            print('Epoch complete in {:.0f}m {:.0f}s'.format(
            epochTimeEnd // 60, epochTimeEnd % 60))
            print('-' * 25)
            
            total_val_loss = 0.0
            correct = 0.0
            for j, valData in enumerate(validLoader, 0):
                val_inputs, val_labels = valData
                #Wrap tensors in Variables
                if use_gpu:
                    val_inputs, val_labels = Variable(val_inputs.float().cuda()), Variable(val_labels.long().cuda())
                else:
                    val_inputs, val_labels = Variable(val_inputs), Variable(val_labels)    

                #Forward pass
                val_outputs = model(val_inputs)
                val_loss_size = criterion(val_outputs, val_labels)
                total_val_loss += val_loss_size.item()
                
                #Accuary
                tmp_acc_val, tmp_acc_index = torch.max(val_outputs, 1)
                #print("val_labels:: ", val_labels)
                #print("tmp_acc_index:: ", tmp_acc_index)
                correct += (tmp_acc_index == val_labels).float().sum()
            
            #print("val_out shape:: {}, val_labels shape {}".format(val_outputs.shape, val_labels.shape))
            #print("val_out :: {}, val_labels:: {}".format(val_outputs, val_labels))
            #print(" len(validLoader) ::\n", len(validLoader))
            val_loss.append(total_val_loss / len(validLoader))
            #Saving the loss over epochs for plotting the graph           
            print("Validation loss = {:.6f}, Accuary = {:.6}".format(total_val_loss / len(validLoader), correct/5000))

            # Plotting Loss vs Epochs
            fig1 = plt.figure(1)        
            plt.plot(range(epoch+1),train_loss,'r--',label='train')        
            plt.plot(range(epoch+1),val_loss,'b--',label='valid')        
            if epoch==0:
                plt.legend(loc='upper left')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
            fig1.savefig('ConvNet_lossPlot.png')             

        time_elapsed = time.time() - training_start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        return model
