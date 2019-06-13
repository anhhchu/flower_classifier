# Imports here


import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim 
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array and a Tensor
    '''
    from PIL import Image
    image = Image.open(image_path)
    
    #resize image
    if image.size[0]> image.size[1]:
        image.thumbnail((100000,256))
    else:
        image.thumbnail((256,100000))
        
    w, h = image.size
    processed_im = image.crop(((w-224)/2,(h-224)/2,(w+224)/2,(h+224)/2))
    
    #convert to numpy array
    np_im = np.array(processed_im)
    
    #normalize
    np_im = (np_im-np.average(np_im))/np.std(np_im)
    np_im = np_im.transpose((2,0,1))
    tensor_im = torch.from_numpy(np_im).float()
    return tensor_im
    # TODO: Process a PIL image for use in a PyTorch model

 #below function shows image using the image tensor input
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def process_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])
    train_datasets = datasets.ImageFolder(train_dir,transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir,transform = valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_datasets,batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_datasets,batch_size = 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_datasets,batch_size = 32, shuffle = True)
    
    return (train_datasets, trainloader, validloader, testloader)

#define the model architecture
def build_model(arch, hidden_units, output_units):
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        input_units = 25088
        
    elif arch == 'vgg19':
        model = models.vgg19(pretrained = True)
        input_units = 25088
        
    elif arch == 'densenet121':
        model = models.densenet121(pretrained = True)
        input_units = 1024
        
    classifier = nn.Sequential(OrderedDict([
        ('fc1',nn.Linear(input_units,hidden_units)), 
        ('relu',nn.ReLU()),
        ('dropout',nn.Dropout(p=0.2)),
        ('fc2',nn.Linear(hidden_units,output_units)),
        ('output',nn.LogSoftmax(dim=1))
    ]))

    #Add the classifier to the pretrained model
    #choose 1 of the 3 models ('densenet121','vgg19','resnet18')
    for params in model.parameters():
            params.requires_grad = False
    model.classifier = classifier

    return model

def load_checkpoint(checkpointfile):
    checkpoint = torch.load(checkpointfile)
    model = build_model(checkpoint['arch'], checkpoint['hidden_units'], checkpoint['output_units'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    #model.to(device)
    model.eval()
    return model




    
    
