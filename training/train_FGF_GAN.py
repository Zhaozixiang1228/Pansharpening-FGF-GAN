# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''


import os
import xlwt
import time
import datetime
import numpy as np

import torch
import torch.nn as nn

from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader 
from scipy.io import savemat

import sys
sys.path.append("..")
from models import get_sat_param
from models.FGF_GAN import GPNN, Discriminator
from metrics import get_metrics_reduced
from utils import PSH5Dataset, PSDataset, prepare_data, normlization, save_param, psnr_loss, ssim


'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''


model_str = 'FGF-GAN'
satellite_str = 'Landsat8'  # QuickBird GaoFen2


# . Get the parameters of your satellite
sat_param = get_sat_param(satellite_str)
if sat_param!=None:
    ms_channels, pan_channels, scale = sat_param
else:
    print('You should specify `ms_channels`, `pan_channels` and `scale`! ')
    ms_channels = 10
    pan_channels = 1
    scale = 2


# . Set the hyper-parameters for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 300
lr = 5e-4
weight_decay = 0
batch_size = 64
n_feat = 32
n_layer = 5
lambda_pixel = 1
lambda_GAN = 1e-2

criterion_GAN = nn.MSELoss().to(device)
criterion_pixelwise = nn.L1Loss().to(device)

# . Get your model 

Tensor = torch.cuda.FloatTensor if device=='cuda' else torch.FloatTensor
net = GPNN(ms_channels,
           pan_channels, 
           n_feat,
           n_layer).to(device)
discriminator = Discriminator(ms_channels*2+pan_channels).cuda()
#print(discriminator)
#print(net)

optimizer_G = torch.optim.Adam(net.parameters(),     lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, [num_epochs//2,num_epochs+1], gamma=0.1)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, [num_epochs//2,num_epochs+1], gamma=0.1)
if satellite_str == 'Landsat8':
    patch = (1, 8, 8)
elif satellite_str == 'QuickBird' or satellite_str == 'GaoFen2':
    patch = (1, 16, 16)


# . Create your data loaders
prepare_data_flag = False # set it to False, if you have prepared dataset
train_path      = '../PS_data/%s/%s_train.h5'%(satellite_str,satellite_str)
validation_path = '../PS_data/%s/validation'%(satellite_str)
test_path       = '../PS_data/%s/test'%(satellite_str)
if prepare_data_flag is True:
    prepare_data(data_path = '../PS_data/%s'%(satellite_str), 
                 patch_size=32, aug_times=1, stride=32, synthetic=False, scale=scale,
                 file_name = train_path)

trainloader      = DataLoader(PSH5Dataset(train_path), 
                              batch_size=batch_size, 
                              shuffle=True) #[N,C,K,H,W]
validationloader = DataLoader(PSDataset(validation_path,scale),      
                              batch_size=1)
testloader = DataLoader(PSDataset(test_path, scale),
                        batch_size=1)
loader = {'train':      trainloader,
          'validation': validationloader}


# . Creat logger
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
save_path = os.path.join(
    'logs/%s'%(model_str),
    timestamp+'_%s'%(satellite_str)
    )
writer = SummaryWriter(save_path)
params = {'model': model_str,
          'satellite': satellite_str,
          'epoch': num_epochs,
          'lr': lr,
          'batch_size': batch_size,
          'n_feat': n_feat,
          'n_layer': n_layer,
          'patch': patch,
          'lambda_pixel': lambda_pixel}
save_param(params,
           os.path.join(save_path, 'param.json'))


'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''


step = 0
best_psnr_val, best_ssim_val, psnr_val, ssim_val =0.,  0., 0., 0.
torch.backends.cudnn.benchmark = True
prev_time = time.time()


for epoch in range(num_epochs):
    ''' train '''
    for i, (ms, pan, gt) in enumerate(loader['train']):
        # 0. preprocess data
        N,C,h,w = ms.shape
        N,c,H,W = pan.shape
        ms, pan, gt = ms.cuda(), pan.cuda(), gt.cuda()
        ms,_ = normlization(ms.cuda())
        pan,_ = normlization(pan.cuda())
        gt,_ = normlization(gt.cuda())
        
        # 1. Create adversarial ground truths
        valid = Variable(Tensor(np.random.rand(N, *patch)/5.+0.9), requires_grad=False)
        fake = Variable( Tensor(np.random.rand(N, *patch)/5.), requires_grad=False)  
        # 2. Train Generators
        net.train()
        net.zero_grad()
        optimizer_G.zero_grad()
        imgf = net(ms, pan) # fake
        discriminator.eval()
        pred_fake = discriminator(imgf, ms, pan)
        loss_GAN = criterion_GAN(pred_fake, valid) # gan loss
        loss_pixel = criterion_pixelwise(imgf, gt) # pixel loss
        loss_G = lambda_GAN * loss_GAN + lambda_pixel * loss_pixel # generator loss
        loss_G.backward()
        optimizer_G.step()
        
        # 3. Train Discriminator
        discriminator.train()
        discriminator.zero_grad()
        optimizer_D.zero_grad()
        pred_real = discriminator(gt, ms, pan)
        loss_real = criterion_GAN(pred_real, valid)# Real loss
        pred_fake = discriminator(imgf.detach(), ms, pan)
        loss_fake = criterion_GAN(pred_fake, fake)# Fake loss
#        loss_D = 0.5 * (loss_real + loss_fake)# Total loss
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()
        
        # 4. print
        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] [PSNR/Best: %.4f/%.4f] [SSIM/Best: %.4f/%.4f] ETA: %s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                psnr_val,
                best_psnr_val,
                ssim_val,
                best_ssim_val,
                time_left,
            )
        )

        # 5. Log the scalar values
        writer.add_scalar('loss/loss3_G_GAN', loss_GAN.item(), step)
        writer.add_scalar('loss/loss2_G_pixel', loss_pixel.item(), step)
        writer.add_scalar('loss/loss1_G', loss_G.item(), step)
        writer.add_scalar('loss/loss4_D', loss_D.item(), step)
        writer.add_scalar('loss/loss6_real', loss_real.item(), step)
        writer.add_scalar('loss/loss5_fake', loss_fake.item(), step)
        writer.add_scalar('learning rate', optimizer_D.state_dict()['param_groups'][0]['lr'], step)
        step+=1
        
    # 4. adjust the learning rate
    scheduler_G.step()#更新学习率 
    scheduler_D.step()   
    
    ''' validation ''' 
    current_psnr_val = psnr_val
    psnr_val = 0.
    ssim_val = 0.
    with torch.no_grad():
        net.eval()
        for i, (ms, pan, gt) in enumerate(loader['validation']):
            ms,_ = normlization(ms.cuda())
            pan,_ = normlization(pan.cuda())
            gt,_ = normlization(gt.cuda())
            imgf = net(ms, pan)
            psnr_val += psnr_loss(imgf, gt, 1.)
            ssim_val += ssim(imgf, gt, 5, 'mean', 1.)
        psnr_val = float(psnr_val/loader['validation'].__len__())
        ssim_val = float(ssim_val/loader['validation'].__len__())
    writer.add_scalar('PSNR on validation data', psnr_val, epoch)
    writer.add_scalar('SSIM on validation data', ssim_val, epoch)

    ''' save model ''' 
    # Save the best weight
#    if best_psnr_val<psnr_val and best_ssim_val<ssim_val:
#    if best_ssim_val<ssim_val:
    if best_psnr_val<psnr_val:    
        best_psnr_val = psnr_val
        best_ssim_val = ssim_val
        torch.save({'G':net.state_dict(),
                    'D':discriminator.state_dict(),
                    'OG':optimizer_G.state_dict(),
                    'OD':optimizer_D.state_dict(),
                    'epoch':epoch}, 
                   os.path.join(save_path, 'best_net.pth'))
    # Save the current weight
    torch.save({'G':net.state_dict(),
                'D':discriminator.state_dict(),
                'OG':optimizer_G.state_dict(),
                'OD':optimizer_D.state_dict(),
                'epoch':epoch}, 
               os.path.join(save_path, 'last_net.pth'))
    
    ''' backtracking '''
    if epoch>0:
        if (best_psnr_val-current_psnr_val)/current_psnr_val<-0.01:
            print(10*'='+'Backtracking!'+10*'=')
            net.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['G'])
            discriminator.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['D'])
            optimizer_D.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['OD'])
            optimizer_G.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['OG'])


'''
------------------------------------------------------------------------------
Test
------------------------------------------------------------------------------
'''


# 1. Load the best weight and create the dataloader for testing
net.load_state_dict(torch.load(os.path.join(save_path, 'best_net.pth'))['G'])



# 2. Compute the metrics
metrics = torch.zeros(5,testloader.__len__())
with torch.no_grad():
    net.eval()
    for i, (ms, pan, gt) in enumerate(testloader):
        ms,_ = normlization(ms.cuda())
        pan,_ = normlization(pan.cuda())
        gt,_ = normlization(gt.cuda())
        imgf = net(ms, pan)
        metrics[:,i] = torch.Tensor(get_metrics_reduced(imgf, gt))
        savemat(os.path.join(save_path,testloader.dataset.files[i].split('\\')[-1]),
               {'HR':imgf.squeeze().detach().cpu().numpy()} )


# 3. Write the metrics
f = xlwt.Workbook()
sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True)
img_name = [i.split('\\')[-1].replace('.mat','') for i in testloader.dataset.files]
metric_name = ['PSNR','SSIM','CC','SAM','ERGAS']
for i in range(len(metric_name)):
    sheet1.write(i+1,0,metric_name[i])
for j in range(len(img_name)):
   sheet1.write(0,j+1,img_name[j])  
for i in range(len(metric_name)):
    for j in range(len(img_name)):
        sheet1.write(i+1,j+1,float(metrics[i,j]))
sheet1.write(0,len(img_name)+1,'Mean')
for i in range(len(metric_name)):
    sheet1.write(i+1,len(img_name)+1,float(metrics.mean(1)[i]))
f.save(os.path.join(save_path,'test_result.xls'))
