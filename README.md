# mhealth-playground

### readData.py

Function `mhealth_get_dataset(dir_to_files = './mhealth-data/')` is a helper function. Which will convert the log files into a **dictionary :** `[ {id:id,data: 2DArrayOfRawData ,file_location:fileLocation},{id:id,data: 2DArrayOfRawData ,file_location:fileLocation}]`

## Ideas (if you have any add them below)

### CNN

Have a CNN which is a X by 23 (number of sensors) image and use a softmax to predict classes.

### Simple neural network

Have a simple neural network which takes X readings from each column and then having a softMax to predict which class it belongs in.

### Simple neural network with LSTM

Same as Simple neural network but with a LSTM. Since the data is temporal (correct me if I am wrong) we can use a LSTM to learn from previous history.

### Simple neural networks concatinated

Have many Simple neural network which try and predict one classifcation and each simple neural network have 1 Dense filter which says how likely it is to be that classification (1-0) (use relu filter https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu ? ). Then concatinate them into a single feed forward network that then predicts the classifiers. Advantages: Maybe faster to train these simple neural networks and then concatinate. Can be reused in a bigger module Disadvantages: I want to train whole network may take more time.
