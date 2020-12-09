# Concrete Critical Crack Width Detection using UNET

Python Implementation of Standard UNET architecture for crack detection through image segmentation as well as post processing for critical area location. Images used are of the mendeley dataset with standardised 256x256 dimensions. Preprocessing is done to guarantee this dimensions therefore your image might be warped when training. 

# Preprocessing used
- Gauss Filter
- N1 Means Denoising
- Histogram Equalisation

# Postprocessing used (crack width)
- Watershed Algorithm
- Skeletonisation

<img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" width="600" height="400" class="center"/>

# Standard functions in main.py
- --tolabel : Uses the specified DIP module to preprocess images into out/tolabel folder
- --dip : Selects the type of digital image processing used (currently only for concrete cracks)
- --measure : Extracts critical crack width from images into out/measure folder
- --train : Runs the specified neural network model for training, output training files will be in out/(dataset name) folder (if not specified, default UNET is used)
- --test : Runs the trained model on a test dataset 
- --gpu: Enables GPU mode
- --augmentation: Augments dataset to create additional images for training or testing purposes

Example:
- python main.py --dataset=example --arch=example --dip=example --gpu --test
- python main.py --dataset=example --dip=example --augmentation=0000
 
# Example Results

<img src="https://github.com/nzpi/UNETConcreteCrack_CriticalWidthDetection/blob/main/output/019_3_original.png?raw=true" width="300" height="300"/>
<img src="https://github.com/nzpi/UNETConcreteCrack_CriticalWidthDetection/blob/main/output/019_4_overlay.png?raw=true" width="300" height="300"/>

Currently the crack detection assumes 0.1mm -> 1 pixel, this can be scaled manually or if an additional reference object is provided in each picture, scaled to that object

# How to use
- Make an src, out and dataset folder
- Copy all github files except output into the src folder
- Make and populate a new train (label and image) and test folder in the dataset folder for your dataset
- If images have not been labelled, to use the dip specified to label the images, label them by running python main.py --dip=example --tolabel with the images in the out/tolabel folder
- Copy the labelled images into the train/label folder
- Run python main.py --dataset=example --arch=example --gpu --train (for training)
- Run python main.py --dataset=example --arch=example --dip=example --gpu --test (for testing)
- Run python main.py --measure (for crack width detection)
