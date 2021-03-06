
# Artificial Neural Networks and Deep Learning Projects

*AN2DL course projects at Politecnico di Milano  
Developed in collaboration with: @Lodz97*

## Homework 1: Image Classification

In order to create our neural network, at first we started with a "custom" implementation. We took inspiration from VGG, and we stacked several convolutional, activation pooling layers. At the end, before the softmax, we used dense layers with dropout to force neurons to learn indipendent features.  
To perform early stopping, we split the dataset in two parts, and we found that 20% for validation was the best compromise.  
We also did some tuning regarding data augmentation (the best parameters are the ones found below) and also for the learning rate; we used a custom callback to change the learning rate at each epoch, as we discovered that this allowed us to better avoid local minima.  
After a few trainings (around 100 epochs), we "hit a wall" with our custom model around 0.65 accuracy in validation, and we decided to try transfer learning.

After creating a custom model, we have moved to the Transfer Learning technique. We have tried different pre-trained models:
* Vgg16
* ResNet152V2
* EfficientNetB7
* EfficientNetB6
* Xception
* InceptionResNetV2

At first we have freezed the weights of the models, and we have added the FC layers composed by a Dense layer with 512 neurons and the output layer. Later, to improve results, we have trained the whole models with the early stopping regularization method; we also performed fine tuning on the learning rate and the data augmentation. Finally we have ensembled the best models by averaging their predictions in order to reduce the overfitting and improve the predictions.

After that we have tried to add another Dense layer in the FC part of the networks with 256 neurons and we have noticed an improvement on the performance especially with EfficientNetB6 and EfficientNetB7. In order to achieve better performance we have tried to reduce the validation set to have a bigger training set, but the networks did not improve their results.  
The final model is composed by the ensemble of: EfficientNetB7 + EfficientNetB6 + Xception + InceptionResNetV2, and can be loaded in the next notebook.

## Homework 2: Image Segmentation

In order to create our neural network, at first we started with a "custom" implementation. For the first part, the encoder that performs the downsampling, we took inspiration from VGG, and we stacked several convolutional, activation pooling layers. Then after the "bottleneck", we have built the decoder part that performs the upsampling, by using convolutional and activation layers with a decreasing number of filters.  
To train the network we performed early stopping techniques on the validation accuracy. We also used data augmentation; we got the best result by using the larger image size that ram allowed, so that the final upsampling of the mask using nearest neighbours had the least possible impact on the result.  
After a few trainings we discovered that our custom model reaches 0.30 IoU in test on Bipbip Mais, and we decided to try transfer learning.

Then we have moved to the Transfer Learning technique for what concerning the encoder part.  
We have tried two different pre-trained models: 
- Vgg16
- Xception 

We have freezed the weights of the models and we have added the decoder part. With the Xception model the IoU was better than the previous one and in test it reaches 0.75. Then we have tried to ensemble the two models by averaging the prediction of each pixel, but the results were a little bit worse.

After that we have tried to implement a U-Net "by hand", by adding the skip connection between the layers but the results were not better than before.  
Finally we have found an implementation of a predefined network similar to U-Net, called Linknet, on github (https://github.com/qubvel/segmentation_models), and we used it for transfer learning. We retrained all the weights and we got a final test IoU of 0.82 on Bipbip Mais. To get the results for all the other teams and crops, we just retrained the model for those datasets without particular tuning.

## Homework 3: Visual Question Answering

For the VQA problem, our approach was to pass the two inputs, the image and the question, to two differents nets and then merge the two latent representations with concatenation to make the final prediction.  
For the image we tried different nets, and at the end InceptionResNetV2 resulted in the best performance.  
For the questions, first we embedded them and then we used a two layers LSTM to extract features.  
After the concatenation we used fully connected layers and a softmax for final prediction. We fine tuned various hyperparameters like embedding size, number of LSTM layers and dropout rate. We had to resize the images so that the entire training set could fit in our RAM.
