import numpy as np
import os
import logging
import SimpleITK as sitk
import scipy
import skimage.transform
from skimage import exposure
import torch
from LASC18.utils import patch_builder, Transformer, distance_map, get_logger


#logger = get_logger('VNetDice_Layers2')

class DatasetGenerator():
    """
    Main dataset generator class for the segmentation model. Input LGE-MRI data of varying sizes are converted to a 96X640X640(DXHXW) array, then
    normalised and augmented(Translation, Horizontal flip, Gaussian noise) using the Transformer class to generate a set of 400 training images.
    Additional augmentations(Rotation, Translation) are performed randomly over every epoch and adds another 80 data points.
    Apart from the randomly generated data, data generated is precomputed and stored for faster access. 
    In the train and validation modes the object returns the augmented raw image, label image tensors(+distance map tensor for HybridLoss).
    In the test mode in addition to image Tensor the object returns raw image filename, original raw image dimensions and slice indices determined from
    the localised  to store final prediction.  
    """
    
    def __init__(self, mode, inputs, pad, scale_factor, loss_criterion = 'Dice', **kwargs):
        """
        Args-
        mode(str): Specify train, test, validation to generate data suited to each specfic mode.
        inputs(torch.Tensor): Inputs from the locator model to localise the Left Atrium structure, consequently determine corresponding slice indices and 
                               appropriate model training patch sizes.
        pad(tuple): additional padding to ensure a buffer for the localised structure. Pipeline uses (30,30,30) 
        scale_factor(tuple): scale factor used by the locator model. Needed to correctly determine localised structure slice indices.
        loss_criterion(str, optional): The loss criterion being used. For HybridLoss the object precomputes the distance maps, provided to the Data Loader.
        
        Kwargs-
        patch_dims(tuple, optional): Provided during validation and test modes to use the same patch sizes used when training the segmentation model.
        mode(float, optional): Provided during validation and test modes to normalise data similar to when training.
        std(float, optional): Provided during validation and test modes to normalise data similar to when training.
        transform(str, optional): Either random_rotate or random_translate.
        seed(int, optional): Seed value to create a specific MT Randomstate object used to generate further seed values at every epoch for implementing
        reproducible random transformations.
        """
        assert mode in ['train', 'test', 'validate'], 'Wrong mode provided. Must be either "train", "test" or "validate"'
        self.loss_criterion = None if mode == 'test' else loss_criterion
        self.mode = mode
        patch_dims = kwargs.get('patch_dims')
        self.patch_size, paths_slices = patch_builder(gen_set_inputs = inputs, pad = pad, scale_factor = scale_factor, patch_size = patch_dims)
        self.random_state = np.random.RandomState(kwargs.get('seed'))        
        self.transform = kwargs.get('transform') if self.mode == 'train' else None        

        if self.mode == 'test':
            self.mean = kwargs.get('mean')
            assert self.mean != None, 'Mean value must be provided'

            self.std = kwargs.get('std')
            assert self.std != None, 'Standard deviation value must be provided'
            
            self.raw_transformed = []
            for data in paths_slices:
                raw_image_path = data[0][0]
                raw_image_file = raw_image_path.split('/')[-1]
                slice_range = data[1]
                raw_image_array = sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(raw_image_path), sitk.sitkUInt8)) # data is from 0 to 255 hence uint8 for efficiency
                raw_depth, raw_height, raw_width = raw_image_array.shape    
                if (raw_depth, raw_height, raw_width) != (96, 640, 640):
                    raw_image_array = np.pad(raw_image_array,
                                             ((int((96 - raw_depth)/ 2), int((96 - raw_depth)/ 2)),
                                              (int((640 - raw_height)/ 2), int((640 - raw_height)/ 2)),
                                              (int((640 - raw_width)/ 2), int((640 - raw_width)/ 2))),
                                             'constant',
                                              constant_values = 0)
                    raw_image_array = raw_image_array[slice_range]
                transformed_raw_image = Transformer(raw_image_array, image_type = 'raw', mode = 'test', mean = self.mean, std = self.std) # mode=test performs only normalisation and Tensor generation
                self.raw_transformed.append((raw_image_file, transformed_raw_image()[0], slice_range, (raw_depth, raw_height, raw_width)))
            
            self.label_transformed = None     

        elif self.mode in ['train', 'validate']:
            self.raw_images = []
            for data in paths_slices:
                raw_image_path = data[0][0]
                raw_image_file = raw_image_path.split('/')[-1]
                slice_range = data[1]
                raw_image_array = sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(raw_image_path), sitk.sitkUInt8))
                raw_depth, raw_height, raw_width = raw_image_array.shape    
                if (raw_depth, raw_height, raw_width) != (96, 640, 640):
                    raw_image_array = np.pad(raw_image_array,
                                             ((int((96 - raw_depth)/ 2), int((96 - raw_depth)/ 2)),
                                              (int((640 - raw_height)/ 2), int((640 - raw_height)/ 2)),
                                              (int((640 - raw_width)/ 2), int((640 - raw_width)/ 2))),
                                             'constant',
                                              constant_values = 0)
                    raw_image_array = raw_image_array[slice_range]                                         
                self.raw_images.append(raw_image_array)
                 
            images_array = np.concatenate([image.ravel() for image in self.raw_images])
            self.mean = kwargs.get('mean') if kwargs.get('mean') != None else np.mean(images_array)
            self.std = kwargs.get('std') if kwargs.get('std') != None else np.std(images_array)
         
        
            self.label_images = []
            for data in paths_slices:
                label_image_path = data[0][1]
                label_image_file = label_image_path.split('/')[-1]
                slice_range = data[1]
                label_image_array = sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(label_image_path), sitk.sitkUInt8))
                label_depth, label_height, label_width = label_image_array.shape    
                if (label_depth, label_height, label_width) != (96, 640, 640):
                    label_image_array = np.pad(label_image_array,
                                             ((int((96 - label_depth)/ 2), int((96 - label_depth)/ 2)),
                                              (int((640 - label_height)/ 2), int((640 - label_height)/ 2)),
                                              (int((640 - label_width)/ 2), int((640 - label_width)/ 2))),
                                              'constant',
                                              constant_values = 0)
                # Convert the label array to binary
                label_image_array[label_image_array == 255] = 1
                label_image_array = label_image_array[slice_range]
                self.label_images.append(label_image_array)
        
            assert len(self.raw_images) == len(self.label_images), 'Unequal number of label files w.r.t to raw files'
        
        
            if self.transform is None: # if random transforms not needed then implement and store precomputed tranforms(400 data points)
                self.raw_transformed = []
                self.label_transformed = []
                for _, images in enumerate(zip(self.raw_images, self.label_images)):
                    raw_image, label_image = images
                    #kernel_size = (raw_image.shape[0] // 3,####
                    #               raw_image.shape[1] // 8,####
                    #               raw_image.shape[2] // 8)####
                    #kernel_size = np.array(kernel_size)####
                    #clip_limit = 0.02####
                    #raw_image = exposure.equalize_adapthist(raw_image, kernel_size = kernel_size, clip_limit = clip_limit)###
                    transformed_raw_image = Transformer(raw_image, image_type = 'raw', mode = self.mode, mean = self.mean, std = self.std)
                    transformed_label_image = Transformer(label_image, image_type = 'label', mode = self.mode) # only normalisation and tensor generation

                    for raw_image in transformed_raw_image():
                        self.raw_transformed.append(raw_image)
                    for label_image in transformed_label_image():
                        self.label_transformed.append(label_image)
              
                if self.loss_criterion == 'HybridLoss':
                    self.maps = []
                    for image in self.label_transformed:
                        self.maps.append(distance_map(image))


        self.len = len(self.raw_transformed) if self.transform is None else len(self.raw_images)
        
    def __getitem__(self, idx):
        if self.mode == 'test':
           
           return self.raw_transformed[idx]

        elif self.transform is None and self.mode != 'test' and self.loss_criterion != 'HybridLoss':
            
            return self.raw_transformed[idx], self.label_transformed[idx], torch.Tensor([])

        elif self.transform is None and self.mode != 'test' and self.loss_criterion == 'HybridLoss':
            
            return self.raw_transformed[idx], self.label_transformed[idx], self.maps[idx]

        elif self.transform is not None and self.loss_criterion != 'HybridLoss': # compute and provide the randomly generating images(80 data points)
            seed = self.random_state.randint(0,10000) # based on the previously created fixed Random State object generate seed values to be used for both label and raw images to ensure both undergo similar tranformations
            raw_random_transformed = Transformer(self.raw_images[idx], image_type = 'raw', mode = self.transform, mean = self.mean, std = self.std, seed = seed)
            label_random_transformed = Transformer(self.label_images[idx], image_type = 'label', mode = self.transform, seed = seed)

            return raw_random_transformed(), label_random_transformed(), torch.Tensor([])     
         
        elif self.transform is not None and self.loss_criterion == 'HybridLoss':
            seed = self.random_state.randint(0,10000)
            raw_random_transformed = Transformer(self.raw_images[idx], image_type = 'raw', mode = self.transform, mean = self.mean, std = self.std, seed = seed)
            label_random_transformed = Transformer(self.label_images[idx], image_type = 'label', mode = self.transform, seed = seed)
            label = label_random_transformed()
            dist_map = distance_map(label)

            return raw_random_transformed(), label, dist_map
 
    def __len__(self):
        
        return self.len


class LocatorDataGenerator():
    """
    Data generator used during Left Atrium localisation. The object reads the input raw and label data file directories performs necessary padding to
    enure training data are all of the same dimensions and then downscales the images for approximate localisation. The object provides only raw and 
    label images during training mode and additionally provides image file paths during testing or when needed by the main Data Generator object(during 
    paths_validate, path_train and test). This object provides the main entry point for the rest of the segmentation pipeline.
    """
    
    def __init__(self, root_dir, mode, scale_factor = (0.5, 0.25, 0.25), **kwargs):
        """
        Args-
        root_dir(str): The root directory containing all the required image files. Folders must be organised into train , test, validate with raw and label
                        within each.
        mode(str): Specify mode to genrate data specific to 'train', 'test' and 'validate' modes of the pipeline. Additional modes consist of 'paths_train'
                    and 'paths_validate' to generate data and their corrseponding file paths to be used by the main Data Generator object. 
        scale_factor(tuple, optional): Scale factor to downsample the input raw and label images for localisation.

        """
        assert mode in ['train', 'test', 'validate', 'paths_train', 'paths_validate'],'Wrong mode provided'
        self.mode = mode
        file = mode.split('_')[-1] if mode.startswith('paths') else mode
        

        raw_image_dir = os.path.join(root_dir, file, 'raw')
        assert os.path.exists(raw_image_dir), "Raw folder doesn't exists within train/test/validate directory. \
                                               Place raw data within raw folder in corresponding directory "
        raw_image_files = os.listdir(raw_image_dir)
        raw_image_files.remove('.DS_Store') if '.DS_Store' in raw_image_files else raw_image_files
        raw_image_files = sorted(raw_image_files)
        if self.mode != 'test':
            label_image_dir = os.path.join(root_dir, file, 'label')
            assert os.path.exists(label_image_dir), "Label folder doesn't exists within train/validate directory. \
                                                    Place label data within label folder in corresponding directory "
            label_image_files = os.listdir(label_image_dir)
            label_image_files.remove('.DS_Store') if '.DS_Store' in label_image_files else label_image_files
            label_image_files = sorted(label_image_files)
        else:
             label_image_files = []   
        

        self.raw_images = []
        for raw_image_file in raw_image_files:
            raw_image_path = os.path.join(raw_image_dir, raw_image_file)
            raw_image_array = sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(raw_image_path), sitk.sitkUInt8))
            raw_depth, raw_height, raw_width = raw_image_array.shape    
            if (raw_depth, raw_height, raw_width) != (96, 640, 640):
                assert all(x <= y for x,y in zip((raw_depth, raw_height, raw_width), (96, 640, 640))),'Cannot perform padding, data larger than expected'
                raw_image_array = np.pad(raw_image_array,
                                         ((int((96 - raw_depth)/ 2), int((96 - raw_depth)/ 2)),
                                          (int((640 - raw_height)/ 2), int((640 - raw_height)/ 2)),
                                          (int((640 - raw_width)/ 2), int((640 - raw_width)/ 2))),
                                          'constant',
                                          constant_values = 0)
            raw_image_array = skimage.transform.rescale(raw_image_array, scale = scale_factor, order = 0, preserve_range = True, anti_aliasing = True)
            raw_image_transformed = Transformer(raw_image_array, image_type = 'raw')
            if self.mode in ['paths_train', 'paths_validate','test']:
                self.raw_images.append((raw_image_path, raw_image_transformed()[0]))
            elif self.mode in ['train', 'validate']:
                self.raw_images.append(raw_image_transformed()[0])
         
        if self.mode == 'test':
            self.label_images = None                
        else:
            self.label_images = []
            for label_image_file in label_image_files:
                label_image_path = os.path.join(label_image_dir, label_image_file)
                label_image_array = sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(label_image_path), sitk.sitkUInt8))
                label_depth, label_height, label_width = label_image_array.shape    
                if (label_depth, label_height, label_width) != (96, 640, 640):
                    assert all(x <= y for x,y in zip((label_depth, label_height, label_width), (96, 640, 640))),'Cannot perform padding, data larger than expected'
                    label_image_array = np.pad(label_image_array,
                                             ((int((96 - label_depth)/ 2), int((96 - label_depth)/ 2)),
                                              (int((640 - label_height)/ 2), int((640 - label_height)/ 2)),
                                              (int((640 - label_width)/ 2), int((640 - label_width)/ 2))),
                                              'constant',
                                              constant_values = 0)
                # Convert the label array to binary
                label_image_array[label_image_array == 255] = 1
                #label_image_array = skimage.transform.rescale(label_image_array, scale = scale_factor, order = 0, preserve_range = False, anti_aliasing= True)
                label_image_array = scipy.ndimage.zoom(label_image_array, zoom = scale_factor, order = 0)
                label_image_transformed = Transformer(label_image_array, image_type = 'label')
                if self.mode in ['paths_train', 'paths_validate']:
                    self.label_images.append((label_image_path, label_image_transformed()[0]))
                elif self.mode in ['train', 'validate']:
                    self.label_images.append(label_image_transformed()[0])
            
            assert len(self.raw_images) == len(self.label_images)
            if self.mode in ['paths_train', 'paths_validate']:
                for _, data in enumerate(zip(self.raw_images, self.label_images)):
                    raw_data, label_data = data
                    raw_file = raw_data[0].split('/')[-1].split('_')[0]
                    label_file =  label_data[0].split('/')[-1].split('_')[0]
                    assert raw_file == label_file, 'Different files provided for raw and label images, ensure same set for both types'


        self.len  = len(self.raw_images)
        
    def __getitem__(self, idx):
        if self.mode == 'test':
           return self.raw_images[idx]

        else:
            return self.raw_images[idx], self.label_images[idx]            
 
    def __len__(self):
        
        return self.len