# Artificial Neural Networks and Deep Learning

<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

<div align="justify">
    
## Overview
This repository contains the jupyter notebooks used to take part at the competitions created for the Artifical Neural Networks and Deep Learning exam at Politecnico di Milano. The covered topics are the following:

* Image Classification
* Semantic Segmentation
* Visual Query Answering

Remember that the notebooks have not the goal to be runned as they are here, but they were runned for the competitions and partially re-runned for producing this repository. If you want to run them, make sure to set up the connections with your Google Drive (Colab was used) and check if the datasets are still avilable for download.


## Image Classification
[![Kaggle](https://img.shields.io/badge/open-kaggle-blue)](https://www.kaggle.com/c/artificial-neural-networks-and-deep-learning-2020/)

The goal of this challenge is to build the best model to solve an image classification problem. The challenge required to classify images depicting groups of people based on the number of masked people. In the specific, solution must discriminate between images depending on the following cases: 
1.  All the people in the image are wearing a mask 
2.  No person in the image is wearing a mask 
3.  Someone in the image is not wearing a mask
    
The dataset consists in 5614 training examples; additional 450 images are provided to compute the final predictions to submit to Kaggle. The evaluation metric is multiclass accuracy.

<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1oKcrfNOhHPXYiTYkqXUgrYMpoZaaIX30" width="400" alt="Masks"/>
</p

To create our neural network, at first we started with a ''custom'' implementation. We took inspiration from VGG, and we stacked several convolutional, activation pooling layers. At the end, before the softmax, we used dense layers with dropout to force neurons to learn independent features. To perform early stopping, we split the dataset in two parts, and we found that 20% for validation was the best compromise. We also did some tuning regarding data augmentation. With our custom model we achieved around 0.65 accuracy in validation, and then we decided to try transfer learning. After creating a custom model, we have moved to the Transfer Learning technique.
We have tried different pre-trained models:

-   Vgg16
-   ResNet152V2
-   EfficientNetB7
-   EfficientNetB6
-   Xception
-   InceptionResNetV2

At first we froze the weights of the models, and then we added the FC layers composed by a Dense layer with 512 neurons and the output layer. Later, to improve results, we have trained the whole models with the early stopping regularization method; we also performed fine tuning on the learning rate and the data augmentation.
Finally we have ensembled the best models by averaging their predictions in order to reduce the overfitting and improve the predictions. After that we have tried to add another Dense layer in the FC part of the networks with 256 neurons and we have noticed an improvement on the performance especially with EfficientNetB6 and EfficientNetB7. In order to achieve better performance we have tried to reduce the validation set to have a bigger training set, but the networks did not improve their results. The final model is composed by the ensemble of: EfficientNetB7 + EfficientNetB6 + Xception + InceptionResNetV2.


## Semantic Segmentation
[![Codalab](https://img.shields.io/badge/open-codalab-green)](https://competitions.codalab.org/competitions/27176) 
  
NOTE: the site that hosted the challenge (Codalab) is changing and we do not know for how long the data will be available.
  
The goal of this challenge is to segment RGB images to distinguish between crop, weeds, and background. The dataset is composed of images captured by different sensors in different moments and are about two kinds of crops: haricot and maize. Data comes from the <a href="http://challenge-rose.fr/en/home/">2019 ROSE Challenge</a> where four teams have competed with agricultural robots. Images in the dataset are divided into different folders based on the team that acquired the image, i.e., Bipbip, Pead, Roseau, Weedelec. For each team, we have two different sub-folders named as the type of crop present in the images, i.e., Haricot and Mais. Training data is provided with ground truth masks. Since this was a two-stage competition, two test sets are provided. The dataset consists in 90 training examples per team per crop, 15 test images per team per crop for the first stage, 15 test images per team per crop for the second stage. The evaluation metric is Intersection over Union (IoU).
<div align="center">
<p float="center">
    <img src="https://drive.google.com/uc?export=view&id=1Dda1f31H5bTHud7zwLjNyMn5OeRjAGGr" width="400" alt="Crop"/>
    <img src="https://drive.google.com/uc?export=view&id=102t3IzXXeQDBl_CtUJN5mNs5_12jkFRS" width="400" alt="Mask"/>
</p>
</div>

To create our neural network, at first we started with a "custom" implementation. For the first part, the encoder that performs the downsampling, we took inspiration from VGG, and we stacked several convolutional, activation pooling layers. Then after the "bottleneck", we have built the decoder part that performs the upsampling, by using convolutional and activation layers with a decreasing number of filters. To train the network we performed early stopping and used data augmentation; we got the best result by using the larger image size that ram allowed, so that the final upsampling of the mask using nearest neighbours had the least possible impact on the result. After a few trainings we discovered that our custom model reached 0.30 IoU in test on Bipbip Mais. 
Then we have moved to the Transfer Learning technique for what concerning the encoder part. We have tried two different pre-trained models:
-	Vgg16
- Xception

We froze the weights of the models and then we added the decoder part. With the Xception model the IoU was better than the previous one and in test it reached 0.75. Finally we have found an <a href="https://github.com/qubvel/segmentation_models">implementation</a> of a predefined network similar to U-Net, called Linknet, and we used it for transfer learning. We retrained all the weights and we got a final test IoU of 0.82 on Bipbip Mais.


## Visual Query Answering 
[![Kaggle](https://img.shields.io/badge/open-kaggle-blue)](https://www.kaggle.com/c/anndl-2020-vqa/)

The goal of this challenge is to solve a visual question answering (VQA) problem on the proposed dataset. The dataset is composed by synthetic scenes, in which people and objects interact, and by corresponding questions, which are about the content of the images. Given an image and a question, the goal is to provide the correct answer. Answers belong to 3 possible categories: 'yes/no', 'counting' (from 0 to 5) and 'other' (e.g. colors, location, ecc.) answers. the dataset is composed of 29333 imaged, 58832 training questions and 6372 test questions. The evaluation metric is multiclass accuracy.

<div align="center">
<p float="center">
    <img src="https://drive.google.com/uc?export=view&id=1llGT5tGbx7qiAnPJmQfTrwH2rWD7b3N8" width="400" alt="Question image 1"/>
    <figcaption>Q: Is the man's shirt blue?  &emsp;A: yes        </figcaption>

</p>
</div>

<div align="center">
<p float="center">
    <img src="https://drive.google.com/uc?export=view&id=1DCgQQTiOWD1tPSGFlZxjrdsUJtA9SUMD" width="400" alt="Question image 2"/>
    <figcaption>    Q: How many bikes?! &emsp; A: 1</figcaption>
</p>
</div>

<p></p>

For the VQA problem, our approach was to pass the two inputs, the image and the question, to two differents nets and then merge the two latent representations with concatenation to make the final prediction. For the image we tried different nets, and at the end InceptionResNetV2 resulted in the best performance. For the questions, first we embedded them and then we used a two layers LSTM to extract features. After the concatenation we used fully connected layers and a softmax for final prediction. We fine tuned various hyperparameters like embedding size, number of LSTM layers and dropout rate. We had to resize the images so that the entire training set could fit in our RAM. We achieved 0.61503 accuracy in test.

## Team
- Luca Colombo [[Github](https://github.com/lucacolombo97)] [[Email](mailto:luca97.colombo@gmail.com)]
- Giacomo Lodigiani [[Github](https://github.com/Lodz97)] [[Email](mailto:giacomo.lodigiani97@gmail.com)]
