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

from LASC18.data import DatasetGenerator, LocatorDataGenerator
from LASC18.utils import *
from LASC18.model import VNet

#os.environ["AWS_DEFAULT_REGION"] = 'eu-wes'
#os.environ["AWS_ACCESS_KEY_ID"] = 'AKIA4FKPSDX'
#os.environ["AWS_SECRET_ACCESS_KEY"] = 'xzi+z0Ct+YfATLsbtgy+xRII/sm'

#s3 = boto3.resource(service_name='s3')

#s3_bucket_name = 'lasc18'

logger = get_logger('VNetAttnHybrid_R4_D4-2')
#logger = get_logger('VNetAttnDice_R4_D4-2')


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
        
        parser.add_argument('--batch_size', dest='batch_size', default=3, type=int,
                          help='batch size for the segmentation model')

        parser.add_argument('--locator_batch_size', dest='locator_batch_size', default=5, type=int,
                          help='batch size for the locator model')

        parser.add_argument('--scale_factor', dest='scale_factor', default=(0.5, 0.25, 0.25), type=tuple,
                          help='scale down factor(D,H,W) for locator model training')
        
        parser.add_argument('--padding', dest='padding', default=(35,35,35), type=tuple,
                          help='padding along each axis for segmentation model inputs')

        parser.add_argument('--locator_learning_rate', dest='locator_lr', default=0.0001, type=float,
                          help='optimizer learning rate for the locator model')

        parser.add_argument('--learning_rate', dest='lr', default=0.00015, type=float,
                          help='optimizer learning rate for the segmentation model')
        
        parser.add_argument('--loss_criterion', dest='loss_criterion', default='Dice', type=str,
                          help='loss function to be used for the segmentation model')
        
        parser.add_argument('--dir_checkpoints', dest='dir_checkpoints', default='/home/xyz/LASC18/Checkpoints',
                          type=str, help='file path to save segmentation state checkpoints')
        
        parser.add_argument('--dir_locator_checkpoints', dest='dir_locator_checkpoints', default='/home/xyz/LASC18/Checkpoints/Locate',
                          type=str, help='file path to save locator state checkpoints')

        parser.add_argument('--best_checkpoints', dest='best_checkpoints', default='/home/xyz/LASC18/Best_Checkpoints',
                          type=str, help='file path to save best performing segmentation state checkpoints')
        
        parser.add_argument('--best_locator_checkpoints', dest='best_locator_checkpoints', default='/home/xyz/LASC18/Best_Checkpoints/Locate',
                          type=str, help='file path to save best performing locator state checkpoints')

        parser.add_argument('--save_after_epochs', dest='save_after_epochs', default=100, type=int,
                          help='number of epochs after which state is saved by default')

        parser.add_argument('--validate_after_epochs', dest='validate_after_epochs', default=1, type=int,
                          help='number of epochs after which validation occurs')
        
        parser.add_argument('--resume', dest='resume', default=None, type=str,
                          help='file path of the stored checkpoint state to resume segmentation training')

        parser.add_argument('--locator_resume', dest='locator_resume', default=None, type=str,
                          help='file path of the stored locator checkpoint state to resume locator training')

        parser.add_argument('--best_locator', dest='best_locator', default='/home/xyz/LASC18/Checkpoints/Locate/24_10 [23:34:57].pt', type=str,
                          help='file path of the best locator checkpoint state to use before segmentation')

        parser.add_argument('--seed', dest='seed', default=123, type=int,
                          help='seed for RNG')

        parser.add_argument('--gpu', action='store_true', dest='gpu', default=True,
                          help='use cuda')

        parser.add_argument('--num_layers', dest='num_layers', default=1, type=int,
                          help='Number of convolution layers in addition to default layers at each level for both models')

        parser.add_argument('--attention_module', dest='attention_module', default=False,
                          help='Use attention mechanism for Segmentation model')

        parser.add_argument('--dilation', dest='dilation', default=1, type=int,
                          help='Dilation in convolution layers for Segmentation model')

        parser.add_argument('--patience', dest='patience', default=7, type=int,
                          help='LR Scheduler patience')

        parser.add_argument('--reduce', dest='reduce', default=0.8, type=float,
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
        locator_batch_size = options.locator_batch_size
        locator_lr_reduce_factor = 0.85
        locator_scheduler_patience = 15
        locator_loss_function = Dice()
        locator_training_error = {'dice_loss': ([],[])}
        locator_validation_error = {'dice_loss': ([],[])}

        best_locator_validation_score = 0
        dir_locator_checkpoints = options.dir_locator_checkpoints
        best_locator_checkpoints = options.best_locator_checkpoints        
        use_cuda = torch.cuda.is_available() and options.gpu == True
        device = torch.device("cuda" if use_cuda else "cpu")
        
        if not os.path.exists(dir_locator_checkpoints):
            os.mkdir(dir_locator_checkpoints)
            
        if not os.path.exists(best_locator_checkpoints):
            os.mkdir(best_locator_checkpoints)    
        
        logger.info(f'''Provided locator checkpoint path: '{dir_locator_checkpoints}'
                                    and best locator checkpoint path: '{best_locator_checkpoints}'
                        ''')
        
        net_locator = VNet(num_layers = 1, dilation = 1)
        logger.info(f'Initialised locator model.')

        locator_optimizer = optim.Adam(net_locator.parameters(), lr = options.locator_lr)
        
        locator_scheduler = lr_scheduler.ReduceLROnPlateau(locator_optimizer, mode = 'min', factor = locator_lr_reduce_factor,
                                                            patience = locator_scheduler_patience, verbose = True)
        
        if options.best_locator is not None:
            assert os.path.isfile(options.best_locator), "Best locator load path provided doesn't exist!"
            best_checkpoint = torch.load(options.best_locator, map_location = 'cpu')
            
            net_locator.load_state_dict(best_checkpoint['locator_model_state'])                
            
            logger.info(f'Loaded best locator model weights.') 

        elif options.best_locator is None and options.locator_resume is not None:
            #s3.Bucket(s3_bucket_name).download_file(Key = '/s3_checkpoints/' + options.resume, Filename = options.resume)
            assert os.path.isfile(options.locator_resume), "Locator resume file path provided doesn't exist!"
            checkpoint = torch.load(options.locator_resume, map_location = 'cpu')
            
            logger.info(f"Loading checkpoint locator model state from resume path: '~/{options.locator_resume}'")    
            
            start_epoch = int(checkpoint['epoch']) + 1
            max_epochs = int(checkpoint['max_epochs'])
            locator_optimizer.load_state_dict(checkpoint['locator_optimizer_state'])
            net_locator.load_state_dict(checkpoint['locator_model_state'])                
            best_locator_validation_score = float(checkpoint['best_locator_validation_score'])
            locator_training_error = checkpoint['locator_training_error']
            locator_validation_error =  checkpoint['locator_validation_error']
            
        elif options.best_locator is None and options.locator_resume is None:
            net_locator.apply(weights_init)
            
            logger.info(f'Initialised model weights.')

        logger.info(f'Number of Trainable parameters for locator model: {number_of_parameters(net_locator)}')

        if use_cuda:
            net_locator.to(device)
            logger.info(f'Training locator with {device} using {torch.cuda.device_count()} GPUs.')

        if options.best_locator is None:
            locator_train_set = LocatorDataGenerator(root_dir = root_dir, scale_factor = options.scale_factor, mode = 'train')
            locator_validation_set = LocatorDataGenerator(root_dir = root_dir, scale_factor = options.scale_factor, mode = 'validate')

            logger.info(f'''Created locator dataset generator objects.
                        ''')

            locator_train_set_loader = DataLoader(locator_train_set, batch_size = locator_batch_size, shuffle = True, num_workers = 4)
            locator_validation_set_loader = DataLoader(locator_validation_set, batch_size = locator_batch_size, shuffle = True, num_workers = 4)

            logger.info(f'''Loaded locator training and validation datasets from '{root_dir}'
                                      Length of locator train set: {len(locator_train_set)} and validation set: {len(locator_validation_set)}
                                      Batch Size: {locator_batch_size}
                                      -----------------------------------------------------------------------------
                                      Beginning locator model training from epoch: {start_epoch} / {max_epochs - 1}
                                      Best locator validation score: {best_locator_validation_score}
                                      Adam optimiser with lr: {'{:.7f}'.format(locator_optimizer.param_groups[0]['lr'])}
                                      Scheduler ReduceLROnPlateau with mode: 'min', factor: {lr_reduce_factor}, patience: {scheduler_patience}
                                      Loss Criterion: 'Dice'
                                      -----------------------------------------------------------------------------
                        ''')


          
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
                
                    locator_training_score['dice_loss'] = torch.cat((locator_training_score['dice_loss'], dice_loss.detach())) if iteration > 0 \
                                                                     else dice_loss.detach()

                locator_train_dice_error = torch.mean(locator_training_score['dice_loss'])
                locator_training_error['dice_loss'][0].append(locator_train_dice_error)
                locator_training_error['dice_loss'][1].append(epoch)

                logger.info(f"Locator training dice error for epoch {epoch} / {max_epochs - 1}:  {locator_train_dice_error}")

                if epoch % options.validate_after_epochs == 0:
                    net_locator.eval()
                    logger.info(f"""-------Performing Validation for locator--------""")

                    with torch.no_grad():
                        for iter, val_data in enumerate(locator_validation_set_loader):
                    
                            val_raw_image, val_label = val_data
                            val_label = val_label.to(device)
                            val_raw_image = val_raw_image.to(device)
                    
                            val_output = net_locator(val_raw_image)       
                    
                            dice_loss = locator_loss_function(val_output, val_label)
                         
                            locator_validation_score['dice_loss'] = torch.cat((locator_validation_score['dice_loss'].cuda(), dice_loss.detach())) if iteration > 0 \
                                                                                else dice_loss.detach()
                
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
                
                    logger.info(f"""Saving locator model state to '{checkpoint_path}'
                                        Locator Training error: {locator_train_dice_error}
                                        Locator Validation error: {locator_validation_dice_error}
                                        Optimizer Learning Rate: {'{:.10f}'.format(locator_optimizer.param_groups[0]['lr'])}
                                        Is Best Locator model: {best_locator_model}
                                """)
                
                    if best_locator_model:
                        best_checkpoint_path = os.path.join(best_locator_checkpoints, t + '.pt')
                        shutil.copyfile(checkpoint_path, best_checkpoint_path)
        
        
        start_epoch = 0
        batch_size = options.batch_size
        lr_reduce_factor = options.reduce
        scheduler_patience = options.patience
        training_error = {'dice_loss': ([],[]), 'focal_loss': ([],[]), 'hybrid_loss': ([],[])}
        validation_error = {'dice_loss': ([],[]), 'focal_loss': ([],[]), 'hybrid_loss': ([],[])}
        best_validation_score = 0
        dir_checkpoints = options.dir_checkpoints
        best_checkpoints = options.best_checkpoints        
        
        loss_criterion = options.loss_criterion
        alpha = 1
        

        assert options.loss_criterion in ['FocalLoss', 'Dice', 'HybridLoss'], 'The specified loss function is not supported!'
        if options.loss_criterion == 'FocalLoss':
            loss_function = FocalLoss()
        elif options.loss_criterion == 'Dice':
            loss_function = Dice()
        elif options.loss_criterion == 'HybridLoss':
            loss_function = HybridLoss()

        if not os.path.exists(dir_checkpoints):
            os.mkdir(dir_checkpoints)
            
        if not os.path.exists(best_checkpoints):
            os.mkdir(best_checkpoints)    
        
        logger.info(f"""Provided checkpoint path: '{dir_checkpoints}'
                                       and best checkpoint path: '{best_checkpoints}""")
        
        net = VNet(num_layers = options.num_layers, dilation = options.dilation, attention_module = options.attention_module)
        logger.info(f'Initialised segmentation model.')    

        optimizer = optim.Adam(net.parameters(), lr = options.lr)
        
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = lr_reduce_factor, patience = scheduler_patience, verbose = True)
        
        if options.resume is not None:
                assert os.path.isfile(options.resume), "Resume file name provided for segmentation model doesn't exist!"
                checkpoint = torch.load(options.resume, map_location = 'cpu')
                
                start_epoch = int(checkpoint['epoch']) + 1
                max_epochs = int(checkpoint['max_epochs'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                net.load_state_dict(checkpoint['model_state'])                
                alpha = int(checkpoint['alpha'])
                best_validation_score = float(checkpoint['best_validation_score'])
                training_error = checkpoint['training_error']
                validation_error =  checkpoint['validation_error']
                
                logger.info(f"Checkpoint segmentation model state loaded from resume path: '~/{options.resume}'")
        else:
            net.apply(weights_init)
            logger.info(f'Initialised segmentation model weights.')
            
        logger.info(f'Number of Trainable parameters for segmentation model: {number_of_parameters(net)}')
        
        if use_cuda:
            net.to(device)
            logger.info(f'Training with {device} using {torch.cuda.device_count()} GPUs.')

        train_set_builder = LocatorDataGenerator(root_dir = root_dir, mode = 'paths_train', scale_factor = options.scale_factor)
        validation_set_builder = LocatorDataGenerator(root_dir = root_dir, mode = 'paths_validate', scale_factor = options.scale_factor)
        train_builder_inputs = []
        validation_builder_inputs = []

        net_locator.eval()
        with torch.no_grad():
            for idx in range(len(train_set_builder)):
                train_raw_data, train_label_data = train_set_builder[idx]
                raw_file_name, raw_image = train_raw_data
                raw_image = torch.unsqueeze(raw_image, dim = 0).to(device)

                train_output = net_locator(raw_image)
                            
                train_builder_inputs.append(((raw_file_name, train_output), (train_label_data)))

            for idx in range(len(validation_set_builder)):
                validate_raw_data, validate_label_data = validation_set_builder[idx]
                raw_file_name, raw_image = validate_raw_data
                raw_image = torch.unsqueeze(raw_image, dim = 0).to(device)

                validate_output = net_locator(raw_image)
                            
                validation_builder_inputs.append(((raw_file_name, validate_output), (validate_label_data)))


        train_set1 = DatasetGenerator(mode = 'train', inputs = train_builder_inputs, pad = options.padding,
                                     scale_factor = options.scale_factor, loss_criterion = options.loss_criterion)

        dataset_mean = train_set1.mean                             
        dataset_std = train_set1.std
        patch_size = train_set1.patch_size

        # Resuming training from checkpoint will affect reproducibility for random rotate and translate sets
        train_set_rotate = DatasetGenerator(mode = 'train', inputs = train_builder_inputs, pad = options.padding,
                                            scale_factor = options.scale_factor, loss_criterion = options.loss_criterion, seed = options.seed,
                                            transform = 'random_rotate', mean = dataset_mean, std = dataset_std, patch_dims = patch_size)

        train_set_translate = DatasetGenerator(mode = 'train', inputs = train_builder_inputs, pad = options.padding,
                                                scale_factor = options.scale_factor, loss_criterion = options.loss_criterion, seed = options.seed,
                                                transform = 'random_translate', mean = dataset_mean, std = dataset_std, patch_dims = patch_size)

        train_set_translate2 = DatasetGenerator(mode = 'train', inputs = train_builder_inputs, pad = options.padding,
                                                scale_factor = options.scale_factor, loss_criterion = options.loss_criterion, seed = options.seed*2,
                                                transform = 'random_translate', mean = dataset_mean, std = dataset_std, patch_dims = patch_size)

        train_set = ConcatDataset([train_set1, train_set_rotate, train_set_translate, train_set_translate2])

        logger.info(f"""Created segmentation train dataset generator objects.
                                        Dataset mean: {dataset_mean}
                                        Dataset std: {dataset_std}
                                        Patch size: {patch_size}""")

        validation_set =  DatasetGenerator(mode = 'validate', inputs = validation_builder_inputs, pad = options.padding,
                                             scale_factor = options.scale_factor, loss_criterion = options.loss_criterion,
                                             mean = dataset_mean, std = dataset_std, patch_dims = patch_size)

                                      
        logger.info(f'Created segmentation validation dataset generator object.')

        train_set_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
        
        validation_set_loader = DataLoader(validation_set, batch_size = batch_size, shuffle = True, num_workers = 8)

        logger.info(f"""Length of train set: {len(train_set)} and validation set: {len(validation_set)}
                                      Batch Size: {batch_size}
                                      -----------------------------------------------------------------------------
                                      Beginning model training from epoch: {start_epoch} / {max_epochs - 1}
                                      Best validation score: {best_validation_score}
                                      Adam optimiser with lr: {'{:.7f}'.format(optimizer.param_groups[0]['lr'])}
                                      Scheduler ReduceLROnPlateau with mode: 'min', factor: {lr_reduce_factor}, patience: {scheduler_patience}
                                      Loss Criterion: '{options.loss_criterion}'
                                      -----------------------------------------------------------------------------
                        """)

        for epoch in range(start_epoch, max_epochs):
            net.train()
            a = {'surface_avg': torch.zeros(1, requires_grad = False, dtype = torch.float32)}
            training_score = {'dice_loss': torch.zeros(1, requires_grad = False, dtype = torch.float32),
                              'focal_loss': torch.zeros(1, requires_grad = False, dtype = torch.float32),
                              'hybrid_loss':torch.zeros(1, requires_grad = False, dtype = torch.float32)} 

            validation_score = {'dice_loss': torch.zeros(1, requires_grad = False, dtype = torch.float32),
                                'focal_loss': torch.zeros(1, requires_grad = False, dtype = torch.float32),
                                'hybrid_loss':torch.zeros(1, requires_grad = False, dtype = torch.float32)} 
            
            logger.info(f"""-------Segmentation training Epoch: [{epoch} / {max_epochs - 1}]-------""")
            
            for iteration, data in enumerate(train_set_loader):
                raw_image_patches, label_patches, dist_maps = data
                raw_image_patches = raw_image_patches.to(device)
                label_patches = label_patches.to(device)

                output_patches = net(raw_image_patches)
            
                if loss_criterion == 'HybridLoss':
                    dice_loss, loss = loss_function(output_patches, label_patches, dist_maps.to(device), alpha)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    training_score['hybrid_loss'] = torch.cat((training_score['hybrid_loss'], loss.detach())) if iteration > 0 else loss.detach()
                    
                elif loss_criterion == 'FocalLoss':
                    dice_loss, loss = loss_function(output_patches, label_patches)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    training_score['focal_loss'] = torch.cat((training_score['focal_loss'], loss.detach())) if iteration > 0 else loss.detach()
                
                else:                   
                    dice_loss = loss_function(output_patches, label_patches)
                    optimizer.zero_grad()
                    dice_loss.backward()
                    optimizer.step()
                
                training_score['dice_loss'] = torch.cat((training_score['dice_loss'], dice_loss.detach())) if iteration > 0 else dice_loss.detach()
                    
            train_dice_error = torch.mean(training_score['dice_loss'])
            train_focal_error = torch.mean(training_score['focal_loss'])
            train_hybrid_error = torch.mean(training_score['hybrid_loss'])

            training_error['dice_loss'][0].append(train_dice_error)
            training_error['focal_loss'][0].append(train_focal_error)
            training_error['hybrid_loss'][0].append(train_hybrid_error)
              
            for loss_name in training_error.keys(): training_error[loss_name][1].append(epoch) 
            
            logger.info(f"""Training dice error for epoch {epoch} / {max_epochs - 1}:  {train_dice_error}""")
            
            net.eval()
            logger.info(f"""-------Performing Validation--------""")##

            with torch.no_grad():#
                for iteration, val_data in enumerate(validation_set_loader):
                    val_raw_image_patches, val_label_patches, val_dist_maps = val_data
                    val_label_patches = val_label_patches.to(device)
                    val_raw_image_patches = val_raw_image_patches.to(device)
                    
                    val_output_patches = net(val_raw_image_patches)
                    
                    if loss_criterion == 'HybridLoss':
                       dice_loss, loss = loss_function(val_output_patches, val_label_patches, val_dist_maps.to(device), alpha)
                       validation_score['hybrid_loss'] = torch.cat((validation_score['hybrid_loss'], loss.detach())) if iteration > 0 else loss.detach()

                    elif loss_criterion == 'FocalLoss':
                         dice_loss, loss = loss_function(val_output_patches, val_label_patches)
                         validation_score['focal_loss'] = torch.cat((validation_score['focal_loss'], loss.detach())) if iteration > 0 else loss.detach()
                    
                    else:       
                        dice_loss = loss_function(val_output_patches, val_label_patches)
                         
                    validation_score['dice_loss'] = torch.cat((validation_score['dice_loss'], dice_loss.detach())) if iteration > 0 else dice_loss.detach()####
                
                validation_hybrid_error = torch.mean(validation_score['hybrid_loss'])
                validation_focal_error = torch.mean(validation_score['focal_loss'])
                validation_dice_error = torch.mean(validation_score['dice_loss'])

                logger.info(f'''Validation dice- epoch {epoch}:  {validation_dice_error}
                            ''')
                
                scheduler.step(validation_dice_error)
                
                
                validation_error['dice_loss'][0].append(validation_dice_error)
                validation_error['focal_loss'][0].append(validation_focal_error)
                validation_error['hybrid_loss'][0].append(validation_hybrid_error)                          
                    
                for loss_name in validation_error.keys(): validation_error[loss_name][1].append(epoch)

            best_model = True if best_validation_score < (1 - validation_dice_error) else False
            
            if best_model:
                best_validation_score = (1 - validation_dice_error)
            
            if best_model or epoch % options.save_after_epochs == 0:
                
                
                if isinstance(net, nn.DataParallel):
                    model_state = net.module.state_dict()
                else:
                    model_state = net.state_dict()
                
                state = {'epoch': epoch,
                         'max_epochs': max_epochs,
                         'optimizer_state': optimizer.state_dict(),
                         'model_state': model_state,
                         'alpha': alpha,
                         'best_validation_score': best_validation_score,
                         'training_error': training_error,
                         'validation_error': validation_error,
                         'patch_size': patch_size,
                         'scale_factor': options.scale_factor,
                         'mean': dataset_mean,
                         'std': dataset_std
                        }
                
                t = time.strftime("%d_%m [%H:%M:%S]", time.localtime())
                checkpoint_path = os.path.join(dir_checkpoints, t + '.pt')
                torch.save(state, checkpoint_path)
                
                logger.info(f'''Saving model state to '{checkpoint_path}'
                                    Training error: {train_dice_error}
                                    Validation error: {validation_dice_error}
                                    Optimizer Learning Rate: {'{:.10f}'.format(optimizer.param_groups[0]['lr'])}
                                    Is Best model: {best_model}
                            ''')
                
                if best_model:
                    best_checkpoint_path = os.path.join(best_checkpoints, t + '.pt')
                    shutil.copyfile(checkpoint_path, best_checkpoint_path)
                    
            if options.loss_criterion == 'HybridLoss' and  (alpha-0.1) >= 0.05:
                    alpha -= 0.01
                    logger.info(f"""
                                        Hybrid Loss alpha value reduced to: {alpha}
                                    """)

        logger.info(f"""
                                      ____________________________________
                                      Finished segmentation model training
                                      ====================================    
                         """)