import logging 
import sys 
import numpy as np 
from torchvision.transforms import Compose 
from LASC18.transforms import * 
import torch 
import torch.nn as nn 
from scipy.ndimage.morphology import distance_transform_edt as distance 
from LASC18.metrics import Distance, JaccardCoefficient, Recall, Precision, Specificity

loggers = {}
def get_logger(name, level = logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(name)s: %(message)s') # message format
        stream_handler.setFormatter(formatter) 
        
        file_handler = logging.FileHandler(f'{name}.log')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        loggers[name] = logger

        return logger            
            
class Transformer(): 
    """
    Provides Transformations applied on the supplied input data. Takes input as numpy array of either raw or label images and along with the specified
    image type and mode necessary to perform appropriate transformation. when an image is supplied with image_type 'label' Normalisation and Gaussian
    noise addition do not take place. The object receives a seed value(same value) for each raw and label image pair supplied by the Data Generator object and
    uses this to perform identical random transformations for both the image types. Currently only random translate and rotate transformations are  possible.
    Changing 'execution_probability' will result in random execution of available transforms based on 'random_state' value created. The object returns 
    a list of CXDXHXW Tensors and during 'test' and 'validation' no transformations are implemented(Normalisation if raw image).
    """
    def __init__(self, image, image_type, mode = 'validate', seed = 123, **kwargs):
        """
        Args-
        image(np.ndarray): Supplied raw or label image by the Data Generator.
        image_type(str): Specify 'raw' or 'label' to perform appropriate transformations
        mode(str, optional): Either 'train', 'test', 'random_rotate', 'random_translate' or 'validate'.
        seed(int, optional): Seed value for producing fixed Random states and consequently reproducible transformations. Value must be same for raw and
                             its corresponding label image to have same transform applied to both to ensure convergence.
        
        Kwargs-
        mean(float, optional): Uses specified  value for normalisation else defaults to normalisation with 0 mean. Must be provided during raw image transformation.
        std(float, optional): Uses specified  value for normalisation else defaults to normalisation with 1 std. Must be provided during raw image transformation.
        """    
        self.image = image
        self.mode = mode
        self.random_state = np.random.RandomState(seed)
        self.mean = kwargs.get('mean') if kwargs.get('mean') != None else 0
        self.std = kwargs.get('std') if kwargs.get('std') != None else 1
        
        normalise = Normalise(mean = self.mean, std = self.std, image_type = image_type)
        horizontal_flip = HorizontalFlip(random_state = self.random_state, execution_probability = 1.0)
        gaussian_noise = GaussianNoise(random_state = self.random_state, image_type = image_type, execution_probability = 1.0)
        rand_rotate = RotateImage(random_state = self.random_state, image_type = image_type, execution_probability = 1.0)
        rand_translate = TranslateImage(random_state = self.random_state, image_type = image_type, execution_probability = 1.0)
        elastic_deformation = ElasticDeformation(random_state = self.random_state, image_type = image_type, execution_probability = 1.0)
        to_tensor = TorchTensor(image_type = image_type)

        if self.mode == 'train': 
            self.transform0 = Compose([normalise, to_tensor])
            self.h_flip = Compose([normalise, horizontal_flip, to_tensor])
            self.g_noise = Compose([normalise, gaussian_noise, to_tensor])
            #self.e_defo = Compose([normalise, elastic_deformation, to_tensor])

        elif self.mode == 'random_rotate':
            self.random = Compose([normalise, rand_rotate, to_tensor])
        
        elif self.mode == 'random_translate':
            self.random = Compose([normalise, rand_translate, to_tensor])

        elif self.mode == 'random_deformation':
            self.random = Compose([normalise, elastic_deformation, to_tensor])    
        
        else:
            self.transform = Compose([normalise, to_tensor])
          
    def __call__(self):
        if self.mode == 'train':
            transformed_images = []
            transformed_images.extend((self.transform0(self.image), self.h_flip(self.image), self.g_noise(self.image)))#, self.e_defo(self.image)))
                                       
            return transformed_images

        elif self.mode in ['random_rotate', 'random_translate', 'random_deformation']:

            return self.random(self.image) # no list returned when random_rotate or random_translate mode
        
        else:

            return [self.transform(self.image)]

# Calculates the number of parameters for a supplied model    
def number_of_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    return sum([np.prod(p.size()) for p in model_parameters])

# Used by Data Generator to define patches that contain mostly the Left atrium structure within raw and label images. This helps reduce the size of data
# consumed by the GPU. 
def patch_builder(gen_set_inputs, pad, scale_factor, patch_size):
    sigmoid = nn.Sigmoid()
    gen_set_ranges = []
    for data in gen_set_inputs:
        raw_image_path, image = data[0]
        label_image_path = data[1][0] if data[1] is not None else None
        image_paths = (raw_image_path, label_image_path) # store image paths so as to provide paths along with corresponding slices at function output 
        image = sigmoid(image).detach().cpu().numpy()
        shape = image[0,0].shape # uses DXHXW
        image_range = []
        for idx, dim_range in enumerate(shape): # identifies the presence of label voxels across each dimension(from the beginning and from the end)
            output = np.rollaxis(image[0,0], idx) # essentially iterates over the available dimensions to identify presence 
            start_index = None
            stop_index = None
            for index in range(dim_range):
                if start_index is None and output[index,:,:].sum() >= 10: # from the beginning
                    start_index = index # store identified start index having label voxels
                if stop_index is None and output[(dim_range - 1) - index,:,:].sum() >= 10: # from the end
                    stop_index = (dim_range - 1) - index # store end index
            assert start_index is not None and stop_index is not None and stop_index > start_index, 'Generated improper indices. Check inputs'
            image_range.append((start_index, stop_index)) 
        gen_set_ranges.append((image_paths, image_range))
        
    max_height = 0
    max_depth = 0
    max_width = 0

    # Calculate the max patch size based on the above identified ranges across all images. Use specified pad to ensure buffer around calculated
    # patch size. Calculated patches are scaled back to original dimensions using specified scale factor.
    # Also calculate unscaled centre coordinates to roughly identify centres of the LA structure to then extract slice ranges later.
    gen_set_centres = []
    for _, data in enumerate(gen_set_ranges):
        image_paths = data[0]
        depth_range, height_range, width_range = data[1]

        depth = round((depth_range[1] - depth_range[0]) / scale_factor[0])
        height = round((height_range[1] - height_range[0]) / scale_factor[1])
        width = round((width_range[1] - width_range[0]) / scale_factor[2])
        
        max_depth = depth if depth > max_depth else max_depth 
        max_height = height if height > max_height else max_height
        max_width = width if width > max_width else max_width

        # calculate the unscaled centre of the structure
        unscaled_centre = (round(depth_range[0] / scale_factor[0]) + round(depth/ 2),
                           round(height_range[0] / scale_factor[1]) + round(height/ 2),
                           round(width_range[0] / scale_factor[2]) + round(width/ 2))
        gen_set_centres.append((image_paths, unscaled_centre))

    max_depth = max_depth + pad[0] if max_depth + pad[0] <= 96 else 96
    max_height = max_height + pad[1] if max_height + pad[1] <= 640 else 640
    max_width = max_width + pad[2] if max_width + pad[2] <= 640 else 640
    
    patch_dimension = patch_size if patch_size is not None else [max_depth , max_height, max_width] # if provided (during testing and validation) use that instead.

    # Modify patch dimensions so as to be suitable with the segmentation model(downsampling across the model)
    for idx, value in enumerate(patch_dimension):
        for _ in range(1,16):
            if value % 16 == 0:
                break
            else:
                value += 1 
        patch_dimension[idx] = value

    image_slices = []
    patch_d = patch_dimension[0] / 2
    patch_h = patch_dimension[1] / 2
    patch_w = patch_dimension[2] / 2

    # calculate the unscaled slice ranges of the centre based on the calculated patch size and LA structure centre
    for data in gen_set_centres:
        paths, centre = data

        # depth slice ranges
        start_depth = centre[0]-patch_d if centre[0]-patch_d > 0 else 0
        end_depth = centre[0]+patch_d if centre[0]+patch_d < 96 else 96
        
        assert end_depth - start_depth <= patch_dimension[0] 
        if end_depth - start_depth != patch_dimension[0]:
          start_depth = 0 if start_depth == 0 else start_depth - (patch_dimension[0] - (end_depth - start_depth))
          end_depth = 96 if end_depth == 96 else end_depth + (patch_dimension[0] - (end_depth - start_depth))
        assert start_depth >= 0 and end_depth <= 96 

        # height slice ranges
        start_height = centre[1]-patch_h if centre[1]-patch_h > 0 else 0
        end_height = centre[1]+patch_h if centre[1]+patch_h < 640 else 640
        
        assert end_height - start_height <= patch_dimension[1] 
        if end_height - start_height != patch_dimension[1]:
          start_height = 0 if start_height == 0 else start_height - (patch_dimension[1] - (end_height - start_height))
          end_height = 640 if end_height == 640 else end_height + (patch_dimension[1] - (end_height - start_height))
        assert start_height >= 0 and end_height <= 640 
         
        # width slice ranges 
        start_width = centre[2]-patch_w if centre[2]-patch_w > 0 else 0
        end_width = centre[2]+patch_w if centre[2]+patch_w < 640 else 640
        
        assert end_width - start_width <= patch_dimension[2] 
        if end_width - start_width != patch_dimension[2]:
          start_width = 0 if start_width == 0 else start_width - (patch_dimension[2] - (end_width - start_width))
          end_width = 640 if end_width == 640 else end_width + (patch_dimension[2] - (end_width - start_width))
        assert start_width >= 0 and end_width <= 640

        image_slice = (slice(int(start_depth), int(end_depth), None),
                       slice(int(start_height), int(end_height), None),
                       slice(int(start_width), int(end_width), None))
        
        image_slices.append((paths, image_slice))
        
    return patch_dimension, image_slices

# class not used
class SegmentationMetrics(nn.Module):
    """
    Object to calculate segmentation metrics between a prediction and groundtruth label.
    """
    def __init__(self, batch_size = 1):
        super(SegmentationMetrics, self).__init__()
        self.dice = Dice()
        self.sigmoid = nn.Sigmoid()
        self.average_dice = 0
        self.image_count = 0
        self.batch_size = batch_size
    
    def calculate_metrics(self, predicted_output, gt_label, mode):
        self.dice_score = self.dice(predicted_output,gt_label).numpy()
        
        if mode != 'train':
            normalised_prediction = self.sigmoid(predicted_output)
            normalised_prediction = normalised_prediction.numpy()
            label = gt_label.numpy()
        
            self.haussdorff_distance = Distance(normalised_prediction, label)
            self.jaccard_similarity = JaccardCoefficient(normalised_prediction, label)
            self.calculated_recall = Recall(normalised_prediction, label)
            self.calculated_precision = Precision(normalised_prediction, label)
            self.calculated_specificity = Specificity(normalised_prediction, label)

    def update_average(self):
        self.average_dice += self.dice_score * self.batch_size
        self.image_count += self.batch_size
        self.average_score =  self.average_dice / self.image_count
        
class Dice(nn.Module):
    """
    Calculates the dice loss between prediction and ground truth label tensors. Prediction tensor must be normalised using sigmoid function before
    calculation. 
    """
    def __init__(self):
        super(Dice, self).__init__()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, predicted_output, label):
        assert predicted_output.size() == label.size(), 'predicted output and label must have the same dimensions'
        predicted_output = self.sigmoid(predicted_output)
        # Resizes or flattens the predicted and label tensors to calculate intersect between them
        predicted_output = predicted_output.view(1, -1)
        label = label.view(1, -1).float()
        intersect = (predicted_output * label).sum(-1)
        denominator = (predicted_output).sum(-1) + (label).sum(-1)
        dice_score = 2 * (intersect / denominator.clamp(min = 1e-6))
        
        return 1.0 - dice_score

class FocalLoss(nn.Module):
    """
        Implements calculation of Focal loss as  FocalLoss(pt) = −(1 − pt)γ log(pt)
        specified in "Lin, T. Y. et al. (2020) ‘Focal Loss for Dense Object Detection’, IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(2), pp. 318–327."
        doi: 10.1109/TPAMI.2018.2858826.
    """
    def __init__(self, gamma = 2, eps = 1e-6, alpha = 1.0, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.dice = Dice()
        self.bce = nn.BCEWithLogitsLoss(reduction = 'none')

    def forward(self, predicted_output, label):
        error = self.dice(predicted_output, label)
        BCE = self.bce(predicted_output, label, reduction = 'none')
        pt = torch.exp(-BCE)
        #focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        focal_loss = (1+(label*199)) * (1 - pt) ** self.gamma * BCE
        
        return error, focal_loss.mean().view(1)



def distance_map(labels) :
    labels = labels.numpy().astype(np.int16)
    assert set(np.unique(labels)).issubset([0,1]), 'Groundtruth labels must only have values 0 or 1'
    result = np.zeros_like(labels) # container to fill in distance values
    for x in range(len(labels)):
        posmask = labels[x].astype(np.bool)
        negmask = ~posmask
        result[x] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask # Level set representation 

    return torch.Tensor(result).to(dtype = torch.int16)

class SurfaceLoss(nn.Module):
    """
    Object to calculate the Surface Loss between a prediction and ground truth image. Based on https://github.com/LIVIAETS/boundary-loss/blob/master/utils.py
    specified in "Kervadec, H. et al. (2018) ‘Boundary loss for highly unbalanced segmentation’, pp. 1–21. doi: 10.1016/j.media.2020.101851."
    Predicted tensor must be normalised using sigmoid function before loss calculation.
    """
    def __init__(self):
        super(SurfaceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, predicted_output, distance_maps) :
        assert predicted_output.shape == distance_maps.shape
        predicted_output = self.sigmoid(predicted_output)
        predicted_output = predicted_output.type(torch.float32)
        batch_surface_loss = predicted_output * distance_maps
        loss = batch_surface_loss.mean()

        return loss.view(1)
    
class HybridLoss(nn.Module):
    """
    Object uses both Dice Loss and Surface Loss in proportions defined in specified parameter alpha to calculate resultant loss to be used for model
    optimisation. (Note: Focal Loss has not been tested but should work.)
    """
    def __init__(self, loss_type = 'Dice', alpha = 1):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.dice = Dice()
        self.loss_1 = Dice() if loss_type == 'Dice' else FocalLoss()
        self.surface_loss = SurfaceLoss()
        
    def forward(self, predicted_output, label, distance_map, alpha):
        self.alpha = alpha
        error = self.dice(predicted_output, label)
        self.dsc = self.alpha * self.loss_1(predicted_output, label)
        self.surface  = (1 - self.alpha) * self.surface_loss(predicted_output, distance_map) 
        return error, self.dsc + self.surface
        #return error, self.alpha * self.loss_1(predicted_output, label) + (1 - self.alpha) * self.surface_loss(predicted_output, distance_map) 