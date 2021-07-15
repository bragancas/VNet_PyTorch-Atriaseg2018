import sys
import os
import shutil
import time
import logging
import argparse
#import boto3
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import numpy as np

from LASC18.data import LocatorDataGenerator
from LASC18.utils import *
from LASC18.model import VNet

# change argument as required
logger = get_logger('VNet__Dice')



def weights_init(net):
    classname = net.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.xavier_normal_(net.weight)
        nn.init.zeros_(net.bias)
    elif classname.find('PRelu') != -1:
        nn.init.xavier_normal_(net.weight)

        

if __name__ == '__main__':
        
        parser = argparse.ArgumentParser()
        
        #parser.add_argument('--root_dir', dest='root_dir', default='/home/ubuntu/LASC18', type=str,
        #                  help='path to the LASC18 dataset')
        parser.add_argument('--root_dir', dest='root_dir', default='/home/xyz/LASC18', type=str,
                          help='path to the LASC18 dataset')

        #parser.add_argument('--s3_bucket_name', dest='s3_bucket_name', default='lasc18', type=str,
        #                  help='AWS S3 bucket to use for checkpoints')
        parser.add_argument('--max_epochs', dest='max_epochs', default=201, type=int,
                          help='number of epochs')
        
        parser.add_argument('--batch_size', dest='batch_size', default=5, type=int,
                          help='batch size')

        parser.add_argument('--scale_factor', dest='scale_factor', default=(0.5, 0.25, 0.25), type=tuple,
                          help='scale down factor(D,H,W) for locator model training')
        
        parser.add_argument('--learning_rate', dest='lr', default=0.0001, type=float,
                          help='optimizer learning rate')
        
        parser.add_argument('--loss_criterion', dest='loss_criterion', default='Dice', type=str,
                          help='loss function to be used')
        
        parser.add_argument('--dir_locator_checkpoints', dest='dir_locator_checkpoints', default='/home/xyz/LASC18/Checkpoints/Locate',
                          type=str, help='file path to save state checkpoints')

        parser.add_argument('--best_locator_checkpoints', dest='best_locator_checkpoints', default='/home/xyz/LASC18/Best_Checkpoints/Locate',
                          type=str, help='file path to save best performing state checkpoints')

        parser.add_argument('--save_after_epochs', dest='save_after_epochs', default=100, type=int,
                          help='number of epochs after which state is saved by default')

        parser.add_argument('--validate_after_epochs', dest='validate_after_epochs', default=1, type=int,
                          help='number of epochs after which validation occurs')
        
        parser.add_argument('--seed', dest='seed', default=123, type=int,
                          help='seed for RNG')
        
        parser.add_argument('--gpu', action='store_true', dest='gpu', default=True,
                          help='use cuda')
        
        parser.add_argument('--num_layers', dest='num_layers', default=1, type=int,
                          help='Number of convolution layers in addition to default layers at each level for locator model')

        parser.add_argument('--attention_module', dest='attention_module', default=False,
                          help='Use attention mechanism for locator model')

        parser.add_argument('--dilation', dest='dilation', default=1, type=int,
                          help='Dilation in convolution layers for Locator model')

        parser.add_argument('--locator_resume', dest='locator_resume', default=None, type=str,
                          help='name of stored model checkpoint state file to resume training')

        parser.add_argument('--best_locator', dest='best_locator', default=None, type=str,
                          help='File path for the finalised best locator model')

        parser.add_argument('--patience', dest='patience', default=15, type=int,
                          help='LR Scheduler patience')

        parser.add_argument('--reduce', dest='reduce', default=0.85, type=float,
                          help='LRScheduler learning_rate reduction factor ')
        
        options = parser.parse_args()
        
        
        #set RNG for both CPU and CUDA
        torch.manual_seed(options.seed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        root_dir = options.root_dir
        #s3_bucket_name = options.s3_bucket_name
        max_epochs = options.max_epochs
        start_epoch = 0
        batch_size = options.batch_size
        loss_criterion = options.loss_criterion
        dir_locator_checkpoints = options.dir_locator_checkpoints
        best_locator_checkpoints = options.best_locator_checkpoints        
        use_cuda = torch.cuda.is_available() and options.gpu == True
        device = torch.device("cuda" if use_cuda else "cpu")
        factor = options.reduce
        patience = options.patience
        
        alpha = 1
        
        if not os.path.exists(dir_locator_checkpoints):
            os.mkdir(dir_locator_checkpoints)
            
        if not os.path.exists(best_locator_checkpoints):
            os.mkdir(best_locator_checkpoints)    
        
        logger.info(f"""Provided locate checkpoint path: '{dir_locator_checkpoints}'
                                    and best locate checkpoint path: '{best_locator_checkpoints}' """)
        
        assert options.loss_criterion == 'Dice', 'Only "Dice" loss function supported! '
        
        loss_function = Dice()
        
        locator_loss_function = Dice()
        best_locator_validation_score = 0
        best_locator_available = False

        locator_training_error = {'dice_loss': ([],[])}
        locator_validation_error = {'dice_loss': ([],[])}


        net_locator = VNet(num_layers = options.num_layers, dilation = options.dilation, attention_module = options.attention_module)
        logger.info(f'Initialised locator model.')

        locator_optimizer = optim.Adam(net_locator.parameters(), lr = options.lr)
        
        locator_scheduler = lr_scheduler.ReduceLROnPlateau(locator_optimizer, mode = 'min', factor = factor, patience = patience, verbose = True)
        
        if options.best_locator is None and options.locator_resume is not None:
            #s3.Bucket(s3_bucket_name).download_file(Key = '/s3_checkpoints/' + options.resume, Filename = options.resume)
            assert os.path.isfile(options.locator_resume), "Locator resume file path provided doesn't exist!"
            checkpoint = torch.load(options.locator_resume, map_location = 'cpu')
                
            start_epoch = int(checkpoint['epoch']) + 1
            max_epochs = int(checkpoint['max_epochs'])
            locator_optimizer.load_state_dict(checkpoint['locator_optimizer_state'])
            net_locator.load_state_dict(checkpoint['locator_model_state'])                
            best_locator_validation_score = float(checkpoint['best_locator_validation_score'])
            locator_training_error = checkpoint['locator_training_error']
            locator_validation_error =  checkpoint['locator_validation_error']
                
            logger.info(f"Checkpoint locator model state loaded from resume path: '~/{options.locator_resume}'")
        
        elif options.best_locator is None and options.locator_resume is None:
            net_locator.apply(weights_init)
            
            logger.info(f'Initialised model weights.')
        
        elif options.best_locator is not None and options.locator_resume is None:
            assert os.path.isfile(options.best_locator), "Best locator load path provided doesn't exist!"
            best_checkpoint = torch.load(options.best_locator, map_location = 'cpu')
            best_locator_available = True

            net_locator.load_state_dict(checkpoint['locator_model_state'])                

        logger.info(f'Number of Trainable parameters: {number_of_parameters(net_locator)}')

        if use_cuda:
            net_locator.to(device)
            logger.info(f'Training using {device} with {torch.cuda.device_count()} GPUs.')

        if not(best_locator_available):
            locator_train_set = LocatorDataGenerator(root_dir = root_dir, mode = 'train', scale_factor = options.scale_factor)
            locator_validation_set = LocatorDataGenerator(root_dir = root_dir, mode = 'validate', scale_factor = options.scale_factor)

            logger.info(f'Created locator dataset generator objects.')

            locator_train_set_loader = DataLoader(locator_train_set, batch_size = batch_size, shuffle = True, num_workers = 4)
            locator_validation_set_loader = DataLoader(locator_validation_set, batch_size = 1, shuffle = True, num_workers = 4)

            logger.info(f"""Loaded locator training and validation datasets from '{root_dir}'
                                      Length of locator train set: {len(locator_train_set)} and validation set: {len(locator_validation_set)}
                                      Batch Size: {batch_size}
                                      -----------------------------------------------------------------------------
                                      Beginning model training from epoch: {start_epoch} / {max_epochs - 1}
                                      Best validation score: {best_locator_validation_score}
                                      Adam optimiser with lr: {'{:.7f}'.format(locator_optimizer.param_groups[0]['lr'])}
                                      Scheduler ReduceLROnPlateau with mode: 'min', factor: {factor}, patience: {patience}
                                      Loss Criterion: 'Dice'
                                      -----------------------------------------------------------------------------
                                        """)


          
            for epoch in range(start_epoch, max_epochs):
                net_locator.train()
                locator_training_score = {'dice_loss': torch.zeros(1, requires_grad = False, dtype = torch.float32)}
                locator_validation_score = {'dice_loss': torch.zeros(1, requires_grad = False, dtype = torch.float32)} 
                logger.info(f"""-------Locator training Epoch: [{epoch} / {max_epochs - 1}]-------""")
                for iteration, data in enumerate(locator_train_set_loader):
                    raw_image, label = data
                    raw_image = raw_image.to(device)
                    label = label.to(device)

                    output = net_locator(raw_image)
                
                    dice_loss = locator_loss_function(output, label)
                
                    locator_optimizer.zero_grad()
                    dice_loss.backward()
                    locator_optimizer.step()
                
                    locator_training_score['dice_loss'] = torch.cat((locator_training_score['dice_loss'], dice_loss.detach())) if iteration > 0 else dice_loss.detach()

                locator_train_dice_error = torch.mean(locator_training_score['dice_loss'])
                locator_training_error['dice_loss'][0].append(locator_train_dice_error)
                locator_training_error['dice_loss'][1].append(epoch)

                logger.info(f'''Locator training dice error for epoch {epoch} / {max_epochs - 1}:  {locator_train_dice_error}''')

                if epoch % options.validate_after_epochs == 0:
                    net_locator.eval()
                    logger.info(f"""-------Performing Validation for locator--------""")

                    with torch.no_grad():
                        for iteration, val_data in enumerate(locator_validation_set_loader):
                            val_raw_image, val_label = val_data
                            val_label = val_label.to(device)
                            val_raw_image = val_raw_image.to(device)
                    
                            val_output = net_locator(val_raw_image)       
                    
                            dice_loss = locator_loss_function(val_output, val_label)
                         
                            locator_validation_score['dice_loss'] = torch.cat((locator_validation_score['dice_loss'], dice_loss.detach())) if iteration > 0 else dice_loss.detach()
                
                    locator_validation_dice_error = torch.mean(locator_validation_score['dice_loss'])
                    locator_validation_error['dice_loss'][0].append(locator_validation_dice_error)    
                    locator_validation_error['dice_loss'][1].append(epoch)
                    logger.info(f"""Val. Dice for Epoch {epoch}:  {locator_validation_dice_error}
                                """)
                    
                    locator_scheduler.step(locator_validation_dice_error)

                best_locator_model = True if best_locator_validation_score < (1 - locator_validation_dice_error) else False

                if best_locator_model:
                    best_locator_validation_score = (1 - locator_validation_dice_error)
            
                if best_locator_model or epoch % options.save_after_epochs == 0:
                    if isinstance(net_locator, nn.DataParallel):
                        model_state = net_locator.module.state_dict()
                    else:
                        model_state = net_locator.state_dict()
                
                    locator_state = {'epoch': epoch,
                                     'max_epochs': max_epochs,
                                     'locator_optimizer_state': locator_optimizer.state_dict(),
                                     'locator_model_state': model_state,
                                     'best_validation_score': best_locator_validation_score,
                                     'locator_training_error': locator_training_error,
                                     'locator_validation_error': locator_validation_error,
                                     'scale_factor': options.scale_factor
                                    }
                
                    t = time.strftime("%d_%m [%H:%M:%S]", time.localtime())
                    checkpoint_path = os.path.join(dir_locator_checkpoints, t + '.pt')
                    torch.save(locator_state, checkpoint_path)
                    #s3.Bucket(s3_bucket_name).upload_file(Filename = checkpoint_path , Key = '/s3_checkpoints/' + t + '.pt')
                
                    logger.info(f"""Saving locator model state to '{checkpoint_path}'
                                       Locator Training error: {locator_train_dice_error}
                                       Locator Validation error: {locator_validation_dice_error}
                                       Optimizer Learning Rate: {'{:.10f}'.format(locator_optimizer.param_groups[0]['lr'])}
                                       Is Best Locator model: {best_locator_model}
                                       """)
                
                    if best_locator_model:
                        best_checkpoint_path = os.path.join(best_locator_checkpoints, t + '.pt')
                        shutil.copyfile(checkpoint_path, best_checkpoint_path)
        logger.info(f"""
                                        _______________________
                                        Finished model training
                                        =======================    
                         """)       