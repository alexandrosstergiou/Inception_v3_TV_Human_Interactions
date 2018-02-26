from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
from skimage import exposure,io
from inceptionv3_highfive_model import model_inceptionv3
from keras import backend as K
from keras.models import model_from_json
from vis.visualization import visualize_cam
from vis.utils import utils
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from vis.visualization import visualize_saliency, overlay

img_path = 'tv_human_interactions_videos/frames/0/0_0034_'

imgs = []

fileextensions = [2,3,4,5,9,10,11,13,14,15,16,17,18,19]

number_of_files = len(fileextensions)

for i in fileextensions:

    path = img_path+str(i)+".png"
    img = image.load_img(path, target_size=(299,299))

    img = image.img_to_array(img)
    img /= 255

    imgs.append(img)

# load model and weiths from json file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('weights.best.hdf5')

X = np.array(imgs, dtype='float32')

"""
#
#VISUALISING ACTIVATIONS CODE
#

print('Image tensor shape: ',img_tensor.shape)

#import matplotlib.pyplot as plt


#plt.imshow(img_tensor[0])
#plt.show()

from keras import models

# Extracts the outputs of the top 29 layers:
layer_outputs = [layer.output for layer in model.layers[:29]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# This will return a list of 5 Numpy arrays:
# one array per layer activation
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[1]
print('First activation layer shape: ',first_layer_activation.shape)

import matplotlib.pyplot as plt
, target_size=(299,299)
#Show 30th channel of the first convlayer
#plt.matshow(first_layer_activation[0, :, :, 30], cmap='magma')
#plt.show()

# Get all layer names
layer_names = []
for layer in model.layers:
    layer_names.append(layer.name)

images_per_row = 16
print('Number of layers: ',len(layer_names))

# Now let's display our feature maps
i = 0
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # Save figure at directory
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    if (i>0):
        plt.imsave("inception_v3_activations_vis/"+"layer_"+str(i)+"_"+layer_name+".png",display_grid,cmap='magma')
    i+=1
    plt.close()
    #plt.imshow(display_grid, aspect='auto', cmap='magma')

#plt.show()

"""


predictions = model.predict(X, verbose=0)

# round predictions
rounded = [round(x[0]) for x in predictions]
string_p = "Predictions for kiss class examples: "+str(rounded)
times = len(string_p)
print("\n"+'\x1b[1;36;40m' +"-"*times+ '\x1b[0m')
print ('\x1b[1;36;40m' +string_p+ '\x1b[0m')
print('\x1b[1;36;40m' +"-"*times+ '\x1b[0m'+"\n")

# This is the "kiss" entry in the prediction vector
kiss_output = model.output[:, 0]

# The is the output feature map of the `conv2d_94` layer,
# the last convolutional layer in Inception v3
last_conv_layer = model.get_layer('conv2d_94')


# This is the gradient of the "kiss" class with regard to
# the output feature map of `conv2d_94`
grads = K.gradients(kiss_output, last_conv_layer.output)[0]

# This is a vector of shape (192,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `conv2d_94`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])


# These are the values of these two quantities, as Numpy arrays,
# given our sample image
#pooled_grads_value, conv_layer_output_value = iterate([imgs])
fig = plt.figure(figsize=(number_of_files,1))
from matplotlib import gridspec
#from PIL import Image
import cv2
gs = gridspec.GridSpec(1,number_of_files)
gs.update(wspace=0.0,hspace=0.0,left=0.1,right=0.9,bottom=0.1,top=0.9)
i = 0
for im in X:
	heatmap = visualize_cam(model, layer_idx=-1, filter_indices=0, seed_input=im, backprop_modifier=None)

	magma_heatmap = np.uint8(cm.magma(heatmap)[..., :3] * 255)
	plt.subplot(gs[i])

	i = i+1
	plt.xticks([])
	plt.yticks([])
	# Overlay is used to alpha blend heatmap onto img.
	#newImage =  cv2.addWeighted(0.5 ,magma_heatmap ,0.5 ,im)
	imagex = (im * 255).round().astype(np.uint8)

	plt.imshow(overlay(magma_heatmap,imagex,alpha=.5))

	plt.axis('off')



plt.show()
