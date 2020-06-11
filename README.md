# mhealth-playground

# Data Science Lifecycle

* 0_Exploring_Data_Structure.ipynb                    (Playing Around)
* 1_Data_Analysis_CorrMatrix_&_DataDist.ipynb	        (Pretty Diagrams)
* 2_Umer_and_Asim_SoftMax_Classifier.py               (Just the same as notebook without some formatting)
* 2_Umer_and_Asim_SoftMax_Classifier_Notebook.ipynb	  (Main Solution)


### ReadData

Function `mhealth_get_dataset(dir_to_files = './mhealth-data/')` is a helper function. Which will convert the log files into a **dictionary :** `[ {id:id,data: 2DArrayOfRawData ,file_location:fileLocation},{id:id,data: 2DArrayOfRawData ,file_location:fileLocation}]`

## Ideas (if you have any add them below)
**UF:** I added some intial ideas but these were before my data exploration.

### CNN

Have a CNN which is a X by 23 (number of sensors) image and use a softmax to predict classes.

### Simple neural network

Have a simple neural network which takes X readings from each column and then having a softMax to predict which class it belongs in.

**UF:** A Multilabel SoftMax classifer. I believe he mentioned he wanted to classify activities so this would output a matrix of 12 activities. Perhaps 13 if there readings also include no activity being done. So the output would be rounded probabilities to 0 or 1 meaning only the highest probability of being an activity would have a 1 (due to SoftMax being relative) and the rest 0. Perhaps we could also show using the probabilities matrix what the 2nd and 3rd highest probabilities would be and for example see if we get jogging and running to be similar high probabilities.

### Simple neural network with LSTM

Same as Simple neural network but with a LSTM. Since the data is temporal (correct me if I am wrong) we can use a LSTM to learn from previous history.

**UF:** This might be a good shout if we are predicting the actual values of the sensors ie a time-series regression performed by an LSTM RNN. Once trained maybe it would take in last 20 values then predict the next value and repeatedly do this (using the predicted value for as history for its next prediction). We would get a pretty good prediction of the sensor of what future sensor reading would be. 

If we are predicting the activity through then time-series classification LSTM model might be the way to go. Something I am not familiar with but may allow us to get way better results than using simple binary classification.

### Simple neural networks concatinated

Have many Simple neural network which try and predict one classifcation and each simple neural network have 1 Dense filter which says how likely it is to be that classification (1-0) (use relu filter https://www.tensorflow.org/api_docs/python/tf/keras/activations/relu ? ). Then concatinate them into a single feed forward network that then predicts the classifiers. Advantages: Maybe faster to train these simple neural networks and then concatinate. Can be reused in a bigger module Disadvantages: I want to train whole network may take more time.

**UF:** Can also test Sigmoid function to see how it holds up for binary classification. We could try binary classification first and show the performance then move onto multivariate SoftMax classification then play around with the rest, play around and compare.

____________________________________________________________________
# Temporary thoughts after some data exploration

I blurted out some initial thoughts as well on that main md but this was before I took a dive into the data. So, I have some more informed ideas now.

I played around with the read data script you made to get some ideas. I then made a separate iPython notebook to see data structure and some messy maths and get an idea of how these targets will work for any models we make. It's quite interesting: for each of the 10 subjects, the last column 23 (0-23 so 24th) range from possible values from 0 to 12 corresponding to each activity with 0 looking like to be no activity I believe. There are 161280 rows/records. Each row has 23 sensor readings (columns 0-22) and then in column 23 (24th column in 0-23) is the classification of what activity is occurring. A classic Multi-label classification problem. 

It will be interesting running this through a SoftMax and seeing how it’ll do.

I did some maths to try and calculate a few things. The subjects clearly tapped on the sensor to let it know which activity is happening. Quick thought I wonder if they had just tapped on it once to change the activity.
Or if they tapped it at the start of the activity, then the sensor does its thing and records at regular intervals and at the end they press it again which causes the last reading. Then theoretically some time may pass until they do the next activity and start it again. Obviously, this could influence the data.

I would think it is the first (most likely) but I haven’t seen this mentioned by the researchers who collected the data (I may have missed something).
