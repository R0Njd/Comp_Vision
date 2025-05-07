
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras import backend as k
from tensorflow.keras.optimizers import  SGD, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

import keras

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

plt.rcParams['figure.figsize'] =[16,10]
plt.rcParams ['font.size'] = 16
from sklearn.model_selection import StratifiedKFold
from mpl_toolkits.axes_grid1 import ImageGrid

# preprocessing steps

SAMPLE_PER_CATEGORIES = 200
SEED =42
WIDTH = 150
HEIGHT =150
DEPTH = 3

INPUT_SHAPE = (WIDTH, HEIGHT, DEPTH)

foldert = "C:\\Users\\ronjd\\OneDrive\\Desktop\\comp_V\\brain_tumor\\Training"
train = os.listdir(foldert)
folder = "C:\\Users\\ronjd\\OneDrive\\Desktop\\comp_V\\brain_tumor\\Testing"
test = os.listdir(folder)

# Define teh catergories 
CATEGORIES =["glioma", "meningioma", "notumor", "pituitary"]
NUM_OF_CATEGORIES = len(CATEGORIES)
print("Number of categories: ", NUM_OF_CATEGORIES)

# check the number of training samples in each category
for category in CATEGORIES:
    print("Number of samples in training  category {}: {}".format(category, len(os.listdir(os.path.join(foldert, category)))))

# check the number of testing samples in each category
for category in CATEGORIES:
    print("Number of samples in testing category {}: {}".format(category, len(os.listdir(os.path.join(folder, category)))))

# resize and convert the image to numpy array
def read_image(file_path, size):
    img = load_img(os.path.join(foldert, file_path), target_size=size)  # load and resize
    img = img_to_array(img) 
    return img

train_list=[]
for category_id, category in enumerate(CATEGORIES):
    for file_name in os.listdir(os.path.join(foldert, category)):
        train_list.append(['{}/{}'.format(category, file_name), category_id, category])
train_list = pd.DataFrame(train_list, columns = ['file', 'category_id', 'category'])
print(train_list.shape)

print(train_list.head())

test_list=[]
for file in test:
   test_list.append(['Testing/{}'.format(file), file])
test_list = pd.DataFrame(test_list, columns = ['filepath' ,'file'])
print(test_list.shape)

print(test_list.head())

#visulaizing the data images 
IMAGE_PER_CATEGORY = 4
fig = plt.figure(1, figsize=(8,8))
grid = ImageGrid(fig, 111, nrows_ncols=(IMAGE_PER_CATEGORY, NUM_OF_CATEGORIES), axes_pad=0.1)    

i = 0
for category_id, category in enumerate(CATEGORIES):
    subset = train_list[train_list['category'] == category]['file'].values[:IMAGE_PER_CATEGORY]
    for filepath in subset:
        ax = grid[i]
        img = read_image(filepath, (WIDTH, HEIGHT))
        ax.imshow(img / 255.0)
        ax.axis('off')
        if i % IMAGE_PER_CATEGORY == IMAGE_PER_CATEGORY - 1:
            ax.text(10, 140, category, bbox=dict(facecolor='white', alpha=0.6))
        i += 1

plt.show()


np.random.seed (seed=SEED)



INPUT_SHAPE = (150, 150, 3)

# Load VGG19 without top FC layers
vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
vgg19.summary()

for i, layer in enumerate(vgg19.layers):
    print(f" {i} {layer.__class__.__name__, layer.trainable}")