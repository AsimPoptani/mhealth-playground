from helper_functions import mhealth_get_dataset
import random
import tensorflow as tf


# Hyperparameters
# This is how many samples per col
data_length = 10

# Choose a random seed for now for reproducibility TODO REMOVE
random.seed(123)

# Get the dataset
dataset=mhealth_get_dataset()

# shuffle dataset
random.shuffle(dataset)

# get 8 training users and 2 test users
training_users,test_users = dataset[:8], dataset[8:]

previous=None
user=training_users[0]['data']
for data in user:
    classifcation=data[23]
    if not previous==classifcation:
        previous=classifcation
        print(classifcation)