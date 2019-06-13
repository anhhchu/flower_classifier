import torch
import numpy as np
from torch import nn
from torch import optim 
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json

from helper import process_data, process_image
#Train the network with pretrainded model, Loop through each epoch and show the running loss for training process
def train_model(data_dir,model,learning_rate, epochs, device):
    device = torch.device("cuda" if torch.cuda.is_available() and device == 'cuda' else "cpu")
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
    epochs = epochs
    
    train_datasets,trainloader, validloader, testloader = process_data(data_dir)
    running_losses, running_valid_losses = [],[]
    for e in range(epochs):
        running_loss = 0
        corrects = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device) #Make sure that the training step is running on gpu

            #clean up accumulated gradients before training the new batch
            optimizer.zero_grad()

            #Forward and backward pass 
            log_ps = model.forward(images)
            loss = criterion(log_ps,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() #add loss of the batch to the running loss

        #use the validation datset to compare train and validation loss    
        else: 
            running_valid_loss = 0
            running_accuracy = 0
            with torch.no_grad():
                model.eval() #set model to evaluation mode to stop dropout 
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)           
                    log_ps = model.forward(images)
                    valid_loss = criterion(log_ps, labels)
                    running_valid_loss += valid_loss.item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy = torch.mean(equals.type(torch.FloatTensor))
                    running_accuracy += accuracy.item()
                    
            model.train()

            print("Epoch:", e+1,
                  "Training loss:{:.3f}..".format(running_loss/len(trainloader)),
                  "Validation loss:{:.3f}..".format(running_valid_loss/len(validloader)),
                  #"Running Accuracy:{:.3f}..".format(running_accuracy),
                  "Validation accuracy:{:.3f}%..".format(running_accuracy*100/len(validloader)))

            running_losses.append(running_loss/len(trainloader))
            running_valid_losses.append(running_valid_loss/len(validloader))
            
    print('Train Losses:',running_losses,
         'Validation Losses:',running_valid_losses)

    return running_losses, running_valid_losses,model


# TODO: Do validation on the test set
def test_data(model, data_dir, device):
    correct = 0
    total = 0
    step = 0
    device = torch.device("cuda" if torch.cuda.is_available() and device == 'cuda' else "cpu")
    model.to(device)
    train_datasets,trainloader, validloader, testloader = process_data(data_dir)
    with torch.no_grad(): #turn off gradient step to reduce computation time and use up resources
        model.eval()
        for images, labels in testloader:
            step += 1
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
            ps = torch.exp(outputs) #convert to softmax probability from 0 to 1 for each image in each batch
            top_p,top_class = ps.topk(1,dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            print('Accuracy for batch',step,':{:.3f}%'.format(accuracy*100))

            correct += sum(equals).item()
            total+= labels.size(0)
    print('Number of correct classified images:',correct)
    print('Number of images in test set:', total)
    print('Accuracy of test set:{:.3f}%'.format(100*correct/total))

def predict(image_path, model, topk,device):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # TODO: Implement the code to predict the class from an image file
        #process the image and show the image  
        model.to(device)
        model.eval()
        tensor_img = process_image(image_path)
        #the model requires the batch size at the start of the input tensor, use unsqueeze to add a dimension of 1 (batch size=1)
        #to dimension 0 of image tensor
        tensor_img.unsqueeze_(0)
        tensor_img.to(device)
        log_results = model.forward(tensor_img)
        probs = torch.exp(log_results)
        top_p, top_class_idx = probs.topk(topk, dim = 1)
        #convert top_p, top_class tensor to numpy and then to list. 
        #Because the variable require Gradient, use var.detach().numpy() instead
        top_probs = top_p.detach().numpy().tolist()[0]
        top_class_idx = top_class_idx.detach().numpy().tolist()[0]
        
        #map the top_class to flower name using model.class_to_idx.items()
        idx_to_class = {val:key for key,val in model.class_to_idx.items()}
        top_class_label = [idx_to_class[label] for label in top_class_idx]
        
        return top_probs, top_class_label
