# Convolutional Pose Machines - Tensorflow

<p align="center">
    <img src="https://github.com/timctho/ConvolutionalPoseMachines-Tensorflow/raw/master/cpm_hand.gif", width="480">
</p>

This is the **Tensorflow** implementation of [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release), one of the state-of-the-art models for **2D body and hand pose estimation**.

## With some additional features:
 - Easy multi-stage graph construction
 - Kalman filters for smooth pose estimation

## Environments
 - Windows 10 / Ubuntu 16.04
 - Tensorflow 1.2.0
 - OpenCV 3.2

## How to start
### Download models
Put downloaded models in the **models/weights** folder.
 - [Body Pose Model](https://drive.google.com/open?id=0Bx1hAYkcBwqnX01MN3hoUk1kUjA)
 - [Hand Pose Model](https://drive.google.com/open?id=0Bx1hAYkcBwqnSU9lSm5Ya3B1VTg)

### Run demo scripts
There are two scripts, **demo_cpm_body.py** for body pose estimation and **demo_cpm_hand.py** for hand pose estimation. 
I take **demo_cpm_hand.py** for example.

First set the **DEMO_TYPE**. If you want to pass an image, then put the path to image here. 
If you want a live demo through a webcam, there are few options. 
 - **MULTI** will show multiple stages output heatmaps and the final pose estimation simultaneously. 
 - **SINGLE** will only show the final pose estimation. 
 - **HM** will show each joint heatmap of last stage separately.

You can also use video files like `.avi`, `.mp4`, `.flv`.

The CPM structure assumes the body or hand you want to estimate is **located in the middle** of the frame.
If you want to avoid that, one way is to add a detector at the begining, and feed the detected bounding box image into this model.

## Build your own model
### Create dataset
See **utils/create_cpm_tfr_fulljoints.py** for an example.
If you want to follow the script, you need to prepare your data like
 - dataset/person_0/imgs/
 - dataset/person_0/labels.txt

And in **labels.txt**, the data format is
`imgs_0.jpg bbox_top_left_y bbox_top_left_x bbox_bot_right_y bbox_bot_right_x joint_0_y joint_0_x joint_1_y joint_1_x ....`

### Training
See **models/nets** for model definition, I take **models/nets/cpm_hand.py** for example.
 - Create a model instance
 - Set how many stages you want the model to have (at least 2)
 - Call **build_loss** if you want to do the training
 - Use **self.train_op** to optimize the model

Please see **train.py** for an example.

