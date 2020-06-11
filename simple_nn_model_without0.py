# I tried removing 0 from classification seems to improve accuracy as expected 1.0. validation accuracy a little bit
# but I haven't fully removed 13 and reduced to 12 classifications.
# For some reason the one hot or the softmax model bit dont like it when I change the values form 13 to 12

from helper_functions import mhealth_get_dataset
import random
import tensorflow as tf
import numpy as np
from collections import defaultdict

# Hyperparameters
# This is how many samples per col
data_length = 2

# Prep data


# Get the dataset
dataset=mhealth_get_dataset()

# shuffle dataset
random.shuffle(dataset)


# training_users,test_data = dataset[:6], dataset[6:]
# training_users,test_data = dataset[:8], dataset[8:] # get 8 training users and 2 test users
training_users,test_data = dataset[:8], dataset[8:]


# Remove all records with a certain value from the list of lists aka the 2D list under the 'data' key in each user dictionary in the list of user information dictionaries.
def remove_records_by_val(lst_of_users_dicos, val=0):
    for i in range(len(lst_of_users_dicos)):
        lst_of_users_dicos[i]['data'] = [lst for lst in lst_of_users_dicos[i]['data'] if lst[23] != val]

#Same thing just not as a function

# for user in training_users:
# for i in range(len(training_users)):
#     training_users[i]['data'] = [lst for lst in training_users[i]['data'] if lst[23] != 0]

# # for user in training_users:
# for i in range(len(test_data)):
#     test_data[i]['data'] = [lst for lst in test_data[i]['data'] if lst[23] != 0]


# In this case removing 0 to see affect on validation accuracy
# The 0 classification (no activity) is very... unpredictable and the network is most certainly having difficulty with this and other activities
remove_records_by_val(training_users, 0)
remove_records_by_val(test_data, 0)



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

def get_one_hot(targets, nb_classes): # 0 taken out means onehot is wrong. 1-12
    # print(targets)
    # print(nb_classes)
    res = np.eye(nb_classes)[np.array(targets - 1).reshape(-1)]
    # res = np.eye(nb_classes)[np.array(targets)]
    # print(res)
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

# removing 0 to see affect on validation accuracy
# labels=get_one_hot(np.array(labels),13)
labels=get_one_hot(np.array(labels),12)

training_labels=labels[:training_labels_len]
test_labels=labels[training_labels_len:]

# Okay lets create our model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(data_length*23), # Our input layer
    tf.keras.layers.Dense(2000,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(2000),

    # tf.keras.layers.Dense(13,activation=tf.keras.activations.softmax) # Our output layer we have 13 classifcations # removing 0 to see affect on validation accuracy
    tf.keras.layers.Dense(12,activation=tf.keras.activations.softmax) # Without 0

])
# Compile our model
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])


model.fit(training_data,training_labels,validation_data=(test_data,test_labels),epochs=200)#100
# -9.2305,3.9479,-4.0236,0.10884,-0.058608,3.4836,-7.467,-8.0885,0.24675,-0.78799,-0.21807,0.19791,18.714,5.4366,2.2399,-5.5648,7.5238,0.47255,-0.25051,1.0172,9.3079,-2.787,-15.309,9
# -8.6677,3.5852,-4.5615,-0.054422,0.22606,-0.75304,-7.1983,-5.9716,0.24675,-0.78799,-0.21807,0.25394,7.7039,9.586,-0.92224,-5.7833,9.6692,0.47255,-0.25051,1.0172,4.6822,1.4321,-15.444,9

# test_array = np.array([[-9.2305,3.9479,-4.0236,0.10884,-0.058608,3.4836,-7.467,-8.0885,0.24675,-0.78799,-0.21807,0.19791,18.714,5.4366,2.2399,-5.5648,7.5238,0.47255,-0.25051,1.0172,9.3079,-2.787,-15.309,-8.6677,3.5852,-4.5615,-0.054422,0.22606,-0.75304,-7.1983,-5.9716,0.24675,-0.78799,-0.21807,0.25394,7.7039,9.586,-0.92224,-5.7833,9.6692,0.47255,-0.25051,1.0172,4.6822,1.4321,-15.444]])

# print(model.predict([test_array])[0][8]*100)




# With 0 

# Epoch 100/100

# training_users,test_data = dataset[:6], dataset[6:] 6 training users 4 test
#157/157 [==============================] - 0s 319us/sample - loss: 0.1574 - acc: 0.9108 - val_loss: 10.5625 - val_acc: 0.3402


#   Without 0_______________________________________________________________________________________________________________________

# 6 training users 4 test
# Epoch 100/100
# 71/71 [==============================] - 0s 479us/sample - loss: 0.0053 - acc: 1.0000 - val_loss: 21.2243 - val_acc: 0.5217

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(data_length*23), # Our input layer
#     tf.keras.layers.Dense(500,activation=tf.keras.activations.relu),
#     tf.keras.layers.Dense(500),
#     tf.keras.layers.Dense(12,activation=tf.keras.activations.softmax) # Without 0
# ])

# 8 training users 2 test
#__________________________

# With Denses 500 in middle
# Epoch 200/200
# 95/95 [==============================] - 0s 790us/sample - loss: 0.0012 - acc: 1.0000 - val_loss: 7.4909 - val_acc: 0.5833

# With Denses in the middle 1000 instead of 500
# Epoch 200/200
# 94/94 [==============================] - 0s 373us/sample - loss: 4.2624e-04 - acc: 1.0000 - val_loss: 20.9304 - val_acc: 0.5833


# With Denses in the middle 2000 instead of 500
# Epoch 200/200
# 95/95 [==============================] - 0s 717us/sample - loss: 2.9468e-04 - acc: 1.0000 - val_loss: 42.0160 - val_acc: 0.6250