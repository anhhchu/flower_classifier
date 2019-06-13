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
import os

from helper import process_data, build_model, imshow
from TrainTestPredictFunc import train_model, test_data 

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification application')
    parser.add_argument(dest="data_dir", default = "flowers",help = "provide directory to image folder, default flowers folder")
    parser.add_argument("-s""--save_dir",dest = "save_dir",action="store", default =os.getcwd()+"/"+"final_checkpoint.pth",help = 'set directory to save checkpoint, default name: final_checkpoint.pth saved on current working directory')
    parser.add_argument("--arch",dest = 'arch', action="store", default='densenet121',help = "choose one of these architectures ['vgg16', 'vgg19', 'densenet121']")
    parser.add_argument("--hidden_units",dest = 'hidden_units', action="store", type=int, default=520, help = "only enter 1 hidden unit; suggested value 4096 if choose vgg network, default values:520 if use densenet121")
    parser.add_argument("--output_units",dest = 'output_units', action="store", type=int, default=102,help = "output_unit default values:102'")
    parser.add_argument("-lr""--learning_rate",dest = 'learning_rate', action="store", type=float, default=0.001,help = "learning rate for optimizer, default 0.001")
    parser.add_argument("-e""--epochs",dest = 'epochs', action="store", type=int, default=5,help = "number of epochs for training, default 5")
    parser.add_argument("-d""--device",dest = 'device', action="store", type=str, default="cuda",help = "device for training,default cuda")
    args = parser.parse_args()
    train_datasets,trainloader, validloader, testloader = process_data(args.data_dir)
    
    model = build_model(args.arch, args.hidden_units,args.output_units)
    running_losses, running_valid_losses, trained_model = train_model(args.data_dir,model,args.learning_rate, args.epochs,args.device)
    test_data(trained_model, args.data_dir, args.device)
    trained_model.class_to_idx = train_datasets.class_to_idx
    #device = torch.to("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    #trained_model.to(device)
    torch.save({'arch':args.arch,
                'hidden_units':args.hidden_units,
                'output_units':args.output_units,
                'state_dict': trained_model.state_dict(),
                'class_to_idx': trained_model.class_to_idx}, args.save_dir)
    
    print("***Use the returned checkpoint path below for prediction:***")
    print(args.save_dir)
    
    
    
    

    






