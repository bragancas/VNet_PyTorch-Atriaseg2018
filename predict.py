import logging
import argparse
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import SimpleITK as sitk

from LASC18.data import DatasetGenerator, LocatorDataGenerator
from LASC18.model import VNet
from LASC18.utils import get_logger


logger = get_logger('Prediction_LASC18')

if __name__ == '__main__':
        
        parser = argparse.ArgumentParser()

        parser.add_argument('--root_dir', dest='root_dir', default='/xyz/LASC18', type=str,
                          help='Directory of the LASC18 dataset')

        parser.add_argument('--locator_path', dest='locator_path', default='None', type=str,
                          help='Best locator path to load the trained locator model before performing testing')
        
        parser.add_argument('--segmentor_path', dest='segmentor_path', default='None', type=str,
                          help='Best segmentation model path')

        parser.add_argument('--gpu', action='store_true', dest='gpu', default=True,
                          help='use cuda')

        parser.add_argument('--num_layers', dest='num_layers', default=1, type=int,
                          help='Number of convolution layers in addition to default layers at each level for both models')

        parser.add_argument('--attention_module', dest='attention_module', default=False,
                          help='Use attention mechanism for Segmentation model')

        parser.add_argument('--dilation', dest='dilation', default=1, type=int,
                          help='Dilation in convolution layers for Segmentation model')

        parser.add_argument('--output_dir', action='output_dir', dest='', type=str,
                          help='Output directory to store the model predictions')

        options = parser.parse_args()

        assert options.locator_path is not None , "Locator load path must be provided during testing mode"
        assert os.path.exists(options.locator_path), "Provided locator load path doesnt exist"
        assert options.segmentor_path is not None, "Segmentation model load path must be provided during testing mode"
        assert os.path.exists(options.segmentor_path), "Provided segmentor load path doesnt exist"
        assert options.root_dir is not None, "Root directory to load test image set must be provided during testing mode"
        #torch.manual_seed(options.seed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        root_dir = options.root_dir
        use_cuda = torch.cuda.is_available() and options.gpu == True
        device = torch.device("cuda" if use_cuda else "cpu")
        
        net_locator = VNet(num_layers = 1, dilation = 1)
        logger.info(f'Initialised Locator model.')
        
        net = VNet(num_layers = options.num_layers, dilation = options.dilation, attention_module = options.attention_module)
        logger.info(f'Initialised segmentation model.')    

        output_dir = os.path.join(root_dir, 'test', 'prediction') if options.output_dir is None else options.output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        logger.info(f"""Provided locator checkpoint load path: '{options.locator_path}'
                                    and segmentor load path: '{options.segmentor_path}' """)
        
        locator_checkpoint = torch.load(options.locator_path, map_location = 'cpu')
        segmentor_checkpoint = torch.load(options.segmentor_path, map_location = 'cpu')

        net_locator.load_state_dict(locator_checkpoint['locator_model_state'])
        net.load_state_dict(segmentor_checkpoint['model_state'])
        patch_size = segmentor_checkpoint['patch_size']
        scale_factor = segmentor_checkpoint['scale_factor']
        mean = segmentor_checkpoint['mean']
        std = segmentor_checkpoint['std']

        logger.info(f"Checkpoint locator model weights loaded from resume path: '~/{options.locator_path}'")
        logger.info(f"Checkpoint segmentation model weights loaded from resume path: '~/{options.segmentor_path}'")
        logger.info(f"Using patch size: {patch_size}, mean:{mean} and std:{std} over the data.")

        if use_cuda:
            net_locator.to(device)
            net.to(device)
            logger.info(f'Testing with {device} using 1 GPUs.')
        

        test_set_builder = LocatorDataGenerator(root_dir = root_dir, mode = 'test', scale_factor = scale_factor)        
        test_builder_inputs = []

        # Roughly locate the LA structure in the test images
        net_locator.eval()
        with torch.no_grad():
            for idx in range(len(test_set_builder)):
                raw_file_name, raw_image = test_set_builder[idx]
                raw_image = torch.unsqueeze(raw_image, dim = 0).to(device)

                train_output = net_locator(raw_image)
                            
                test_builder_inputs.append(((raw_file_name, train_output), None))

        # Use localisation prediction for patch slice generation and to build final test image tensors
        test_set = DatasetGenerator(mode = 'test', inputs = test_builder_inputs, pad = (30,30,30), scale_factor = scale_factor,
                                    mean = mean, std = std, patch_dims = patch_size)

        net.eval()
        with torch.no_grad():
            sigmoid = nn.Sigmoid()
        	for index in range(len(test_set)):
        		raw_image_filename, raw_image, slice_range, output_dims = test_set[index]
                raw_image = torch.unsqueeze(raw_image, dim = 0).to(device)
        		
                prediction = net(raw_image)
				
                prediction = sigmoid(prediction)
                prediction[prediction < 1] = 0
                prediction = prediction[0,0].to(dtype = torch.uint8).detach().cpu().numpy()
                
                prediction_helper = np.zeros((96,640,640), dtype = np.uint8)
                prediction_helper[slice_range] = prediction

                # Gather slice ranges to recreate original input data dimension for each predicted array
                depth_diff = prediction_helper.shape[0] - output_dims[0]
                depth_start = int(depth_diff/ 2)
                depth_end = int(prediction_helper.shape[0] - depth_diff/ 2)

                height_diff = prediction_helper.shape[1] - output_dims[1]
                height_start = int(height_diff/ 2)
                height_end = int(prediction_helper.shape[1] - height_diff/ 2)

                width_diff = prediction_helper.shape[2] - output_dims[2]
                width_start = int(width_diff/ 2)
                width_end = int(prediction_helper.shape[2] - width_diff/ 2)

                # Select from the prediction array to create final output similar to original image arrays
                prediction = prediction_helper[(slice(depth_start, depth_end, None),
                                                slice(height_start, height_end, None),
                                                slice(width_start, width_end, None)
                                                )]


                prediction_image = sitk.GetImageFromArray(prediction)
                prediction_file_path = os.path.join(output_dir, 'prediction_' + raw_image_filename)
                sitk.WriteImage(prediction_image, prediction_file_path)
                logger.info(f"Saved prediction for file: {raw_image_filename} to path: {prediction_file_path} ")