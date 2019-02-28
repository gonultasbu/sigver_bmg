#%% [markdown]
# # Using a feature representation learned for signature images
# 
# This notebook contains code to pre-process signature images and to obtain feature-vectors using the learned feature representation on the GPDS dataset

#%%
import numpy as np

# Functions to load and pre-process the images:
from scipy.misc import imread, imsave
from preprocess.normalize import normalize_image, resize_image, crop_center, preprocess_signature

# Functions to load the CNN model
import signet
from cnn_model import CNNModel

# Functions for plotting:
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.rcParams['image.cmap'] = 'Greys'

#%% [markdown]
# ## Pre-processing a single image

#%%
original = imread('data/some_signature.png')


#%%
# Manually normalizing the image following the steps provided in the paper.
# These steps are also implemented in preprocess.normalize.preprocess_signature

normalized = 255 - normalize_image(original, size=(952, 1360))
resized = resize_image(normalized, (170, 242))
cropped = crop_center(resized, (150,220))


#%%
# Visualizing the intermediate steps

f, ax = plt.subplots(4,1, figsize=(6,15))
ax[0].imshow(original, cmap='Greys_r')
ax[1].imshow(normalized)
ax[2].imshow(resized)
ax[3].imshow(cropped)

ax[0].set_title('Original')
ax[1].set_title('Background removed/centered')
ax[2].set_title('Resized')
ax[3].set_title('Cropped center of the image')

#%% [markdown]
# ## Processing multiple images and obtaining feature vectors

#%%
user1_sigs  = [imread('data/a%d.png' % i) for i in  [1,2]]
user2_sigs  = [imread('data/b%d.png' % i) for i in  [1,2]]

canvas_size = (952, 1360)

processed_user1_sigs = np.array([preprocess_signature(sig, canvas_size) for sig in user1_sigs])
processed_user2_sigs = np.array([preprocess_signature(sig, canvas_size) for sig in user2_sigs])


#%%
# Shows pre-processed samples of the two users

f, ax = plt.subplots(2,2, figsize=(10,6))
ax[0,0].imshow(processed_user1_sigs[0])
ax[0,1].imshow(processed_user1_sigs[1])

ax[1,0].imshow(processed_user2_sigs[0])
ax[1,1].imshow(processed_user2_sigs[1])

#%% [markdown]
# ### Using the CNN to obtain the feature representations

#%%
# Path to the learned weights
model_weight_path = 'models/signet.pkl'


#%%
# Instantiate the model
model = CNNModel(signet, model_weight_path)


#%%
# Obtain the features. Note that you can process multiple images at the same time

user1_features = model.get_feature_vector_multiple(processed_user1_sigs, layer='fc2')
user2_features = model.get_feature_vector_multiple(processed_user2_sigs, layer='fc2')

#%% [markdown]
# ### Inspecting the learned features
# 
# The feature vectors have size 2048:

#%%
user1_features.shape


#%%
print('Euclidean distance between signatures from the same user')
print(np.linalg.norm(user1_features[0] - user1_features[1]))
print(np.linalg.norm(user2_features[0] - user2_features[1]))


#%%
print('Euclidean distance between signatures from different users')

dists = [np.linalg.norm(u1 - u2) for u1 in user1_features for u2 in user2_features]
print(dists)


#%%
# Other models:
# model_weight_path = 'models/signetf_lambda0.95.pkl'
# model_weight_path = 'models/signetf_lambda0.999.pkl'

#%% [markdown]
# # Using SPP models (signatures from different sizes)
# 
# For the SPP models, we can use images of any size as input, to obtain a feature vector of a fixed size. Note that in the paper we obtained better results by padding small images to a fixed canvas size, and processed larger images in their original size. More information can be found in the paper: https://arxiv.org/abs/1804.00448

#%%
from preprocess.normalize import remove_background

# To illustrate that images from any size can be used, let's process the signatures just 
# by removing the background and inverting the image

normalized_spp = 255 - remove_background(original)

plt.imshow(normalized_spp)


#%%
# Note that now we need to use lists instead of numpy arrays, since the images will have different sizes. 
# We will also process each image individually

processed_user1_sigs_spp = [255-remove_background(sig) for sig in user1_sigs]
processed_user2_sigs_spp = [255-remove_background(sig) for sig in user2_sigs]


#%%
# Shows pre-processed samples of the two users

f, ax = plt.subplots(2,2, figsize=(10,6))
ax[0,0].imshow(processed_user1_sigs_spp[0])
ax[0,1].imshow(processed_user1_sigs_spp[1])

ax[1,0].imshow(processed_user2_sigs_spp[0])
ax[1,1].imshow(processed_user2_sigs_spp[1])


#%%
import signet_spp_300dpi
# Instantiate the model
model = CNNModel(signet_spp_300dpi, 'models/signet_spp_300dpi.pkl')


#%%
# Obtain the features. Note that we need to process them individually here since they have different sizes

user1_features_spp = [model.get_feature_vector(sig, layer='fc2') for sig in processed_user1_sigs_spp]
user2_features_spp = [model.get_feature_vector(sig, layer='fc2') for sig in processed_user2_sigs_spp]


#%%
print('Euclidean distance between signatures from the same user')
print(np.linalg.norm(user1_features_spp[0] - user1_features_spp[1]))
print(np.linalg.norm(user2_features_spp[0] - user2_features_spp[1]))


#%%
print('Euclidean distance between signatures from different users')

dists = [np.linalg.norm(u1 - u2) for u1 in user1_features_spp for u2 in user2_features_spp]
print(dists)


