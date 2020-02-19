# RARE EVENT DETECTION:

This repository uses two methods to make a rare-event classifier. Both of them use reconstruction losses from the input data.

The dataset used in the repository can be downloaded from the link given here. https://docs.google.com/forms/d/e/1FAIpQLSdyUk3lfDl7I5KYK_pw285LCApc-_RcoC0Tf9cnDnZ_TWzPAw/viewform


Courtesy of ---   Dataset: Rare Event Classification in Multivariate Time Series - Chitta Ranjan, Mahendranath Reddy, Markku Mustonen, Kamran Paynabar, Karim Pourak
https://arxiv.org/abs/1809.10717


1. The first model 'ae_rse' uses a autoencoder for reconstruction loss. The cases where the reconstrucion loss is very high is treated as anamoly(in our case a break in the system).

2. The second model uses LSTM layers to regenerate the models simpy without using the encoder layer. This gives us a better accuracy as compared to the last case.

Please refer: https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098
for better understanding.

The best roc-auc score of 0.78 is achieved using specific threshold value.

The confusion matrix for the same is shown below.

## How to use the model.
First and foremost download the dataset from the link given above.

## Training
Make the directory structure as shown below:


Run train.py to which will generate the model for both modes ae_rse and rse modes in separate directories.
Or you can also download the model from the google drive link provided here which contains the entire output folders for both the case.

### Testing
To test the model run the file - predict.py it will create a confusion matrix of the labels. Note you must have the data renamed as rare_events.csv inside the data folder to run the predict.py file.

### Results
Below are my results:
I achieved a roc_auc_score of 0.78 from the rse model and a roc_auc_score of 0.75 with the threshold values of 0.165 and 0.175 respectively.

The confusion matrix for the same are shown below:
#### 1. AE_RSE model
##### Tensorboard logs:
##### confusion matrix:



#### 1. RSE model
##### Tensorboard logs:
##### confusion matrix:




