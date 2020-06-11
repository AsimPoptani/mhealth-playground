from helper_functions import mhealth_get_dataset
import random
import tensorflow as tf
import numpy as np
from collections import defaultdict

# Hyperparameters
# This is how many samples per col
data_length = 1

# Prep data


# Get the dataset
dataset=mhealth_get_dataset()

# shuffle dataset
random.shuffle(dataset)

# get 6 training users and 4 test users
training_users,test_data = dataset[:6], dataset[6:]

training_data_pre  = defaultdict(list) 
test_data_pre =defaultdict(list)

counter=0
previous_label=None
to_put=[]
for user in training_users:
    for user_data in user['data']:
        if not previous_label==user_data[23]:
            counter=1
            to_put=[]
            previous_label=user_data[23]
            user_data.pop()
            to_put+=user_data
        elif previous_label==user_data[23]:
            counter+=1
            user_data.pop()
            to_put+=user_data
        if counter == data_length:
            training_data_pre[previous_label].append(to_put)
            to_put=[]

counter=0
previous_label=None
to_put=[]
for user in test_data:
    for user_data in user['data']:
        if not previous_label==user_data[23]:
            counter=1
            to_put=[]
            previous_label=user_data[23]
            user_data.pop()
            to_put+=user_data
        elif previous_label==user_data[23]:
            counter+=1
            user_data.pop()
            to_put+=user_data
        if counter == data_length:
            test_data_pre[previous_label].append(to_put)
            to_put=[]

training_data,training_labels=[],[]
test_data,test_labels=[],[]

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

for training_label,training in training_data_pre.items():
    for train in training:
        training_labels.append(int(training_label))
        training_data.append(train)

training_data=np.array(training_data)


for training_label,training in test_data_pre.items():
    for train in training:
        test_labels.append(int(training_label))
        test_data.append(train)

test_data=np.array(test_data)

training_labels_len=len(training_labels)
labels=training_labels+test_labels
labels=get_one_hot(np.array(labels),13)
training_labels=labels[:training_labels_len]
test_labels=labels[training_labels_len:]

# Okay lets create our model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(data_length*23), # Our input layer
    tf.keras.layers.Dense(500,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(500),
    tf.keras.layers.Dense(13,activation=tf.keras.activations.softmax) # Our output layer we have 13 classifcations

])
# Compile our model
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(training_data,training_labels,validation_data=(test_data,test_labels),epochs=100)
# -9.2305,3.9479,-4.0236,0.10884,-0.058608,3.4836,-7.467,-8.0885,0.24675,-0.78799,-0.21807,0.19791,18.714,5.4366,2.2399,-5.5648,7.5238,0.47255,-0.25051,1.0172,9.3079,-2.787,-15.309,9
# -8.6677,3.5852,-4.5615,-0.054422,0.22606,-0.75304,-7.1983,-5.9716,0.24675,-0.78799,-0.21807,0.25394,7.7039,9.586,-0.92224,-5.7833,9.6692,0.47255,-0.25051,1.0172,4.6822,1.4321,-15.444,9
# test_array = np.array([[-9.2305,3.9479,-4.0236,0.10884,-0.058608,3.4836,-7.467,-8.0885,0.24675,-0.78799,-0.21807,0.19791,18.714,5.4366,2.2399,-5.5648,7.5238,0.47255,-0.25051,1.0172,9.3079,-2.787,-15.309,-8.6677,3.5852,-4.5615,-0.054422,0.22606,-0.75304,-7.1983,-5.9716,0.24675,-0.78799,-0.21807,0.25394,7.7039,9.586,-0.92224,-5.7833,9.6692,0.47255,-0.25051,1.0172,4.6822,1.4321,-15.444]])

# print(model.predict([test_array])[0][8]*100)

# data_length = 1
# Epoch 100/100
# 157/157 [==============================] - 0s 287us/sample - loss: 0.2621 - acc: 0.9569 - val_loss: 0.6089 - val_acc: 0.9144