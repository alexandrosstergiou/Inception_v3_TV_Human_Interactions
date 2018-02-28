# Inception V3 for TV Human Interactions dataset
Applying Transfer Learning on Inception V3 model (weights trained on Imagenet) for the Oxford TV Human Interactions dataset. The network gets as inputs images extracted every 5 frames from videos.

![alt text](https://github.com/alexandrosstergiou/Inception_v3_TV_Human_Interactions/blob/master/inception_v3_activations_vis/layer_1_conv2d_1.png "Layer_1_Conv_2D")

**Activations from the first convolutional layer that handles the input image**

![alt text](https://github.com/alexandrosstergiou/Inception_v3_TV_Human_Interactions/blob/master/images/Grad-cam-kiss.png "Grad_cam")

**Grad-cam for the kiss class of an example from the HighFive dataset**

## Installation
Git is required to download and install the repo. You can open Terminal (for Linux and Mac) or cmd (for Windows) and follow these commands:
```sh
$ sudo apt-get update
$ sudo apt-get install git
$ git clone https://github.com/alexandrosstergiou/Inception_v3_TV_Human_Interactions.git
```

## Dependencies
The network was build with Keras while using the TensorFlow backend.  `scikit-learn` was used as a supplementary package for doing a train-validation split. Additionally, for the grad-cam visualisations the [`keras-vis`](https://github.com/raghakot/keras-vis) toolkit was employed. Considering a correct configuration of Keras, to install the dependencies follow:
```sh
$ sudo pip install -U scikit-learn
$ sudo pip install keras-vis
```

## License
MIT


## Contact
Alexandros Stergiou

a.g.stergiou at uu.nl

Any queries or suggestions are much appreciated!
