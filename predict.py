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
cwd = os.getcwd()
import json
from helper import process_image, build_model, load_checkpoint
from TrainTestPredictFunc import predict
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification application')
    parser.add_argument(dest="image_path", default ="flowers/test/28/image_05230.jpg" ,action="store",help = "provide path to image, the path should look similar to this 'flowers/test/28/image_05230.jpg'")
    parser.add_argument(dest = "checkpoint",default = "ImageClassifier/final_checkpoint.pth", action="store",help = "paste in the checkpoint saved from training, default: ImageClassifier/final_checkpoint.pth")
    parser.add_argument("--top_k",dest = "top_k",default = 5, type=int, action="store",help = "how many most likely classes to display, default = 5")
    parser.add_argument("--device",dest = 'device', action="store", default='cpu',help = "device for prediction,default cpu")
    parser.add_argument("-c""--category_names",dest = "category_names",default = "cat_to_name.json", action="store",help = "the file to map categories to real names, default: 'cat_to_name.json' located on the working directory")
                        
    args = parser.parse_args()
    loaded_model = load_checkpoint(args.checkpoint)
    top_probs, top_class_label = predict(args.image_path, loaded_model, args.top_k, args.device)
    max_probs = max(top_probs)
    
    im_tensor = process_image(args.image_path)
    with open(args.category_names, 'r') as f:
         cat_to_name = json.load(f)
    img_label = cat_to_name[args.image_path.split('/')[-2]]
    print('***Prediction Results***')
    print('*top_probs:', top_probs)
    print('*top_class_label"', top_class_label)
    top_flowers = [cat_to_name[label] for label in top_class_label]
    print('*top_flowers:', top_flowers)
    print("*max class probability:", top_probs[0],
          "*predicted class label:", top_class_label[0],
          "*predicted flower:", top_flowers[0])
    print("*True Image class label:", args.image_path.split('/')[-2], "-"
        "*True Image label:", img_label)

      
        
    
   
