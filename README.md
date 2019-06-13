# Image-Classifier-Application-with-Pytorch

This project aims to train an image classifier to recognize different species of flowers, then build a command line application to train the model and predict any dataset of labled images.

The dataset has 102 flower species: this dataset. You can download this dataset to work on your local machine. However the training step requires Cuda GPU for faster processing time 

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

## The files in this repo:
1. Image Classifier Application Project using Pytorch.ipynb : The project file in Jupyter Notebook where most steps are conducted and validated 
2. train.py and predict.py: 2 main python functions to run the command line application. 
3. Helper functions:
    * helper.py : Host functions to process images, view images, save and load checkpoint of the model
    * TrainTestPredictFunc.py : Host functions to train, validate and predict images    

    
