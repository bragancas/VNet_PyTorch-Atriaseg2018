# Pytorch_VNet_atriaseg2018

An image segmentation project using *PyTorch* to segment the Left Atrium(LA) in 3D Late gadolinium enhanced - cardiac MR images*(LGE-CMR)* of the human heart. 

Cardiovascular diseases are responsible for over 3.8 million deaths in Europe  with high populations dying as a result of CVD it is necessary to diagnose CVD and some ways are throught bio markers however they take a lot of time for manual delineation and so this

A baseline network architecture based on the [2D UNet](https://arxiv.org/abs/1505.04597), the [*VNet*](https://doi.org/10.1109/3DV.2016.79) which uses volumetric convolutional kernels is trained initially and then enhanced using [*attention modules*](http://bmvc2018.org/contents/papers/0092.pdf) to observe performance gain. Both networks are trained using *Dice loss* and a hybrid of *Dice* + [*Boundary loss*](https://doi.org/10.1016/j.media.2020.101851) in an attempt to reduce the average Hausdorff Distance(HD) between the predicted and ground truth(GT) atrial structure boundaries.

For some more depth on
- [Dataset](link)
- [LA Anatomy and Atrial fibrillation](link)
- [Biomedical Image segmentation](link)

## References
[V-Net: Fully convolutional neural networks for volumetric medical image segmentation](https://doi.org/10.1109/3DV.2016.79) Milletari, F., Navab, N. and
Ahmadi, S. A.

[Automatic 3D Atrial Segmentation from GE-MRIs Using Volumetric Fully Convolutional Networks](https://doi.org/10.1007/978-3-030-12029-0_23) Xia, Q. et al.

[Boundary loss for highly unbalanced segmentation](https://doi.org/10.1016/j.media.2020.101851) Kervadec, H. et al. 

[BAM: Bottleneck attention module](http://bmvc2018.org/contents/papers/0092.pdf) Park, J. et al. (BMVC 2018)

## Dependencies
- Python (v3.6.8)
- PyTorch (v1.6.0 + CUDA v10.1)
- Torchvision (v0.7.0 + CUDA v10.1)
- Numpy (v1.18.5)
- Scikit-image (v0.17.2)
- SciPy (v1.4.1)
- SimpleITK (v2.0.0)

## Input Data
The dataset consists of 100 3D MRIs having dimensions of either 88x576x576(DxHxW) or 
88x640x640(DxHxW) voxels and are available in *nrrd* format with each having a spatial resolution of 0.625x0.625x0.625 mm<sup>3</sup>. The input data voxels are grayscale intensities with values from 0-255 and the label data voxels are binary with value of 0 or 255(converted to 1) representing background or atrial structure respectively. This dataset is split into 80 and 20 images for training and validation respectively.

<img align="left" src="https://github.com/bragancas/VNet_PyTorch-Atriaseg2018/blob/master/raw15.gif">

<img height="288" width="288" align="center" src="https://github.com/bragancas/VNet_PyTorch-Atriaseg2018/blob/master/raw15_2.gif">


## Implementation Details

The attached link shows the training + testing piplines and network architecture≈ being implemented in this project.

[Training and testing pipeline depiction ](https://github.com/bragancas/VNet_PyTorch-Atriaseg2018/blob/master/Implementation%20Details.pdf)

[Network architecture](https://github.com/bragancas/VNet_PyTorch-Atriaseg2018/blob/master/Network%20schematic.pdf)

### Data pre-processing
The images for each data point is converted from nrrd into a numpy array using SimpleITK. They are then zero padded to 96x640x640 to eliminate variation in image dimensions (aswell as to account for an even feature map progression through the VNet architecture stages). Viewing the images it's observed that the atrial structure we intend to segment occupies a very small portion(on average 20%) of an entire image with most of the voxels comprising of background. The background bears no significance on defining the atrial structure and eliminating it would reduce data handling and training times drastically. To achieve this, for each dimension the start and end indexes/slices of the image in which the atrial structure is present must be learned, allowing the volume containing the structure to be cropped.

 A baseline network(locator) is trained using downscaled datapoints[scaled by 0.5x0.25x0.25(DxHxW)]. Downscaling compromises the atrial structure details and consequently accuracy, but is sufficient to learn the approximate indexes to crop the pertinent structure volume. A suitable buffer(later determined to be 30x30x30) is added to the indexes which widens the volume under consideration to account for detail lost when performing downsampling. The final calculated indexes(learnt+buffer) are then upscaled back to initial dimensions and is used to crop patches containing the meaningful atrial volume. The resulting cropped patch is fed into the mentioned network architectures for accurate segementation.

 Note: After determining indexes for all the images, the pair of indexes which yields the largest volume/patch is selected as the overall patch size of image input to the network and consequently the indexes for all the images are modified to produce patches having the same size as the largest patch. This is done to provide input of uniform dimension to the segmentation network. Amongst other factors the patch size mainly varies depending on the amount of buffer used. Currently a patch of dimension 96x144x224 is being used.


### Data Augmentation
The training images are normalised, with the following augmentations performed individually on them
* Horizontal flip {80 images}
* Additive noise from a gaussian distribution with 0 mean 1 std {80 images}
* On the fly random rotation between -3 to 3 degrees along depth dimension {80 images} 
* Two sets of on the fly random translation performed along either one or two axes (H or W or H+W) using combinations of -5 and/or 5 voxels [(0,5,0) , (0,0,5) , (0,-5,0) , (0,-5,5) , (0,...)] {160 images}

The above generated images in addition to the initial images yields a total of 480 images that is used for training.

### Network Architecture

<img align="left" src="https://github.com/bragancas/VNet_PyTorch-Atriaseg2018/blob/master/Network%20schematic.pdf">

### Hyperparameters selected
* 5x5x5 Convolutions with stride 1
* 2x2x2 Convolutions with stride 2 for VNet downsampling
* 2x2x2 Transposed Convolutions with stride 2 for VNet upsampling
* PReLU non linearity

* Locator network batchsize = 5
* Segmentation network batchsize = 3
* Max epochs = 200
* Adam optimiser using learning rates- 
  - Locator with Dice Loss = 0.0001  
  - Segmentation with Dice/Hybrid Loss = 0.0003  
  - Segmentation with Dice/Hybrid Loss and attention modules = 0.00025
  - Segmentation with increased convolutional filters = 0.00015
* Locator network Learning rate scheduler with patience = 15 and reduction factor = 0.85
* Segmentation networks Learning rate scheduler with patience = 7 and reduction factor = 0.8
* Reduction ratio = 4 and dilation ratio = 3 or 4 for the attention module

### Loss Functions

#### <ins>Dice Loss</ins>
Based on the Dice score which measures the similarity or the amount of overlap between two given volumes. Its value ranges from 0 to 1 where 1 represents total similarity. The Dice loss is then calculated as 1 - Dice Score, which is to be minimised. Dice score has various formulations and here we use,

&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; **Dice Loss** = 1 - (2 * p * g) / (p + g + ε)
					  
where, p<sub>i</sub> is the predicted binary volume   

&emsp; &emsp; &ensp; g<sub>i</sub> is the ground truth volume
 
&emsp; &emsp; &ensp; ε is used to avoid zero division


Note: Ideally a weighted formulation of the Dice loss would have to be used owing to the imbalance between having more background 
voxels compared to foreground. This is avoided in the current implementation as data pre-processing levels this imbalance, although
weighting could still be  implemented for good measure.

#### <ins>Focal Loss</ins>
This loss function builds on binary cross entropy loss by using a scalable modulating factor to down-weight the contribution of easily classififed
voxels towards the gradient calculation and focus on the harder to classify voxels(having low value of p<sub>t</sub>). Focal loss is defined as:

&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; **Focal loss** = - α<sub>t</sub> (1-p<sub>t</sub>)<sup>γ</sup> log(p<sub>t</sub>)

where, (1-p<sub>t</sub>)<sup>γ</sup> is the modulating factor with γ ≥ 0 as a focusing parameter  

&emsp; &emsp; &ensp; &ensp;p<sub>t</sub> = {p 	&ensp; &ensp; if y = 1

&emsp; &emsp; &emsp; &ensp; &ensp; &ensp;{1-p&ensp; if y = 0

here, y denotes the ground truth class and p the predicted probability of the foreground class

&emsp; &ensp; &ensp;α<sub>t</sub> ∈ [0,1] is the weighting factor


Note: Although this loss function is integrated into the pipeline, the models performance when using it hasn't been observed. 

#### <ins>Boundary Loss</ins>
Boundary loss uses a precomputed level set function on the space containing predicted and ground truth voxels as a distance metric and weights the networks voxel predictions. This allows predicted voxels to be considered along with information on their distance to the ground truth boundary. This is contrary to popular regional losses such as Dice which measures the region/volume of overlapping voxels but where all misclassified voxels are treated equally.

&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; **Boundary loss** = <sub><img src="https://github.com/bragancas/texttest/blob/master/Boundary_loss.png"></sub>


where,	Ω denotes the spatial volume under consideration and q ∈ Ω

&emsp; &ensp; &ensp; &ensp;φ<sub>G</sub>(q) denotes the level set representation of the GT boundary, used to calculate distance from predictions to GT

&emsp; &ensp; &ensp; &ensp;s<sub>θ</sub>(q) ∈ [0,1] denotes the output predictions 

#### <ins>Hybrid loss</ins> (Dice/Focal + Boundary Loss)
To avoid scenarios such as when there are no foreground predictions or when there is no overlap, which might cause training to get stuck Boundary loss is used in conjunction with Dice loss to initially establish a meaningful volume of overlap. This is carried out by scheduling the boundary loss to take greater precedence after a threshold Dice score has been attained or towards latter epochs of training. 

**NOTE:** The current model implementation outputs logits and will require the output tensors to be normalised appropriately, e.g. by applying Sigmoid or Softmax.

### Evaluation metrics
* Dice Score                          																													 

* Jaccard index(or IoU) -                               																								 

 &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; **IoU(a,b)** = |a **∪** b| / |a **∩** b|

* Precision - Precision is defined as the fraction of identified instances that are relevant                                                     

 &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; **Precision** = True Positives/(True Positives + False Positives)

* Recall - Sensitivity or recall is the percentage of correctly identified true positives                                                                

 &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; **Recall** = True Positives/(True Positives + False Negatives)

* Hausdorff Distance - The Hausdorff Distance measures the maximum bi-directional Euclidean distance between two contours                                

 &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; **Hausdorff Distance** = max[max(dist<sub>**a→b**</sub>),max(dist<sub>**b→a**</sub>)]


### Data post-processing
Once the segmentation output of the cropped volume is determined, an empty volume(96x640x640 of np.zeros) is created and the previously saved cropping indexes are used to insert the segmentation output within the empty volume. Subsequently, the zero padding is removed and the volume is converted back to nrrd format to attain the final predicted segmentation volumes for each corresponding input MRI. 

Note: The post-processing step is only performed in the predict.py script as it's not needed in the model training script/phase of the pipeline.

## Output Data

<img align="left" src="https://github.com/bragancas/VNet_PyTorch-Atriaseg2018/blob/master/mri+label15.gif">
<img align="center" src="https://github.com/bragancas/VNet_PyTorch-Atriaseg2018/blob/master/prediction15.gif">


<img height="288" width="288" align="left" src="https://github.com/bragancas/VNet_PyTorch-Atriaseg2018/blob/master/mri+label15_2.gif">
<img height="288" width="288" align="center" src="https://github.com/bragancas/VNet_PyTorch-Atriaseg2018/blob/master/prediction15_2.gif">


## Results

|Architecture|No. of Parameters| Training time | Loss Fn. | Batch size | LR | Haus. dist.(mm) | IoU | Precision | Recall | Dice Score |
|    :---:    |        :---:      |      :---:    |   :---:  |    :---:   |:---:|    :---:   |:---:|    :---:  |  :---: |   :---: |
| VNet (Locator)      				|10.75 M|1 Hr   |Dice  |5|0.0001  |          -    | 0.623 ± 0.166 | 0.855 ± 0.128 | 0.7 ± 0.2    | 0.750 ± 0.175  |
| VNet       		 				|10.75 M|14.5 Hr|Dice  |3|0.0003  |21.77 ± 14.263 | 0.788 ± 0.1   | 0.903 ± 0.077 | 0.857 ± 0.89 | 0.877 ± 0.073  |
| VNet       		 				|10.75 M|14.5 Hr|Hybrid|3|0.0003  |16.886 ± 12.99 | 0.785 ± 0.101 | 0.911 ± 0.078 | 0.850 ± 0.091 | 0.875 ± 0.072 |
| VNet + Attention (redn.=4, dil.=3)|10.75 M|18 Hr  |Dice  |3|0.00025 |17.458 ± 10.715| 0.788 ± 0.090 | 0.907 ± 0.079 | 0.856 ± 0.076 | 0.878 ± 0.063 |
|VNet + Attention (redn.=4, dil.=4)	|10.75 M|18 Hr  |Dice  |3|0.00025 |20.983 ± 11.928| 0.800 ± 0.085 | 0.913 ± 0.071 | 0.863 ± 0.071 | 0.886 ± 0.061 |
|VNet + Attention (redn.=4, dil.=4)	|10.75 M|18 Hr  |Hybrid|3|0.000025|18.048 ± 12.26 | 0.733 ± 0.127 | 0.892 ± 0.124 | 0.803 ± 0.100 | 0.838 ± 0.099 |
|VNet (Increased conv. filters)	   	|16.2 M |26.5 Hr|Dice  |3|0.00015 |20.472 ± 12.224| 0.796 ± 0.075 | 0.921 ± 0.052 | 0.857 ± 0.080 | 0.884 ± 0.051 |

## Training and Testing commands
An instance of a training command using Dice loss and without attention modules on a single GPU whose ID=0:

```zsh
CUDA_VISIBLE_DEVICES=0 python3 train.py --batch_size 3 --loss_criterion 'Dice' \
					--learning_rate 0.0002 --patience 7 \
					--reduce 0.8 --num_layers 1 \
					--save_after_epochs 100 \
					--attention_module False --dilation 1
```

Similarly for testing the above trained model use the same parameter values and run:

```zsh
CUDA_VISIBLE_DEVICES=0 python3 predict.py --num_layers 1 --attention_module False \
					  --dilation 1
```

**Note:** Before training or testing, the appropriate directories for dataset access, saving/loading checkpoints and saving model predictions, etc are specified in the *default* parameter of the corresponding *ArgumentParser argument* within train.py or predict.py.

## Side Notes
### Training
1. The location and segmentation phases were performed separately with the path for best performing locator model supplied when executing the train.py script for segmentation. 
2. Training Dice scores was observed over 30 epochs using learning rates within a range of 0.001-0.00001 in gradual decrements. The highest lr value showing stable increase in performance over the epochs was selected.
3. To avoid interpolation artifacts, rotation augmentation is limited to 3 degrees.
4. To reduce GPU idle times during augmentation computations the flip and additive noise augmentations were pre-computed.
5. Elastic deformation augmentation is supported but not included as part of the pipeline as it's compute intense(more so for 3D arrays). It can be included in the augmentation pre-computation(`self.e_defo`) or can be included by instantiating a Dataset generator object having argument transform = 'random_deformation' and including it in the ConcatDataset object.
6. The training pipeline loads the dataset as well as the augmentation computations into RAM instead of computing them every iteration. This is inorder to avoid GPU idle times and necessitates a large available RAM memory(dependent on number of augmentations used). 
7. GPU used was TitanV having 12 GB memory.

### Outcomes

### Useful packages and Software
1. [3D Slicer](https://www.slicer.org)
2. [pynrrd](https://github.com/mhe/pynrrd)
3. [napari](https://github.com/napari/napari)
4. [TorchIO](https://github.com/fepegar/torchio)
5. [NiBabel](https://github.com/nipy/nibabel)

### Reproducibility
A predefined seed value ensures reproducibility(aspects apart from user defined hyperparameters) in data split generation, initialising model weights and biases(via torch.manual_seed) and generating random on the fly augmentations. However, it must be noted that if the training process is stopped midway and restarted using saved checkpoints, the order in which the seed values are generated and provided to create random on the fly augmentations(via `seed = self.random_state.randint(0,10000)`) will also have restarted and thus will affect reproducibility.
