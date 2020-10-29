# Let's implement 3d sknet in UCF101
from pathlib import Path
import json
import random
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
import torchvision

from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from dataset import get_training_data, get_validation_data, get_inference_data
from utils import Logger, worker_init_fn, get_lr
from training import train_epoch
from validation import val_epoch
import inference

import time

#--------------------------------------------------------
from utils import get_normalize_method, get_mean_std, get_opt


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)

"""
Input : opt and model parameters (for optimizer)
Returns : train_loader, train_sampler, train_logger, train_batch_logger, optimizer, scheduler
"""
def get_train_utils(opt, model_parameters):
    # spatial transform ----------------------
    assert opt.train_crop in ['random', 'corner', 'center']
    # random
    spatial_transform = []
    if opt.train_crop == 'random':
        spatial_transform.append(
            RandomResizedCrop(
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales))
    elif opt.train_crop == 'center':
        spatial_transform.append(Resize(opt.sample_size))
        spatial_transform.append(CenterCrop(opt.sample_size))
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    # horizontal flip
    if not opt.no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    if opt.colorjitter:
        spatial_transform.append(ColorJitter())
    spatial_transform.append(ToTensor())
    # X. we only use rgb
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.append(ScaleValue(opt.value_scale))
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    # temporal transform ---------------------------
    assert opt.train_t_crop in ['random', 'center']
    temporal_transform = []
    if opt.sample_t_stride > 1:        # default = 1
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':   # default = random
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    # train data ---------------------------
    train_data = get_training_data(opt.video_path, opt.annotation_path,
                                   opt.dataset, opt.input_type, opt.file_type,
                                   spatial_transform, temporal_transform)
    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    # loader
    print("Is shuffle", (train_sampler is None))
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn,
                                               drop_last=True)

    if opt.is_master_node:
        train_logger = Logger(opt.result_path / 'train.log',
                              ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            opt.result_path / 'train_batch.log',
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    else:
        train_logger = None
        train_batch_logger = None

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    
    optimizer = SGD(model_parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov)

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.plateau_patience)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer, opt.multistep_milestones)
    
    return (train_loader, train_sampler, train_logger, train_batch_logger, optimizer, scheduler)


def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model


def resume_train_utils(resume_path, begin_epoch, optimizer, scheduler):
    print('loading checkpoint {} train utils'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    begin_epoch = checkpoint['epoch'] + 1
    print("begin_epoch", begin_epoch)
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return begin_epoch, optimizer, scheduler

def get_val_utils(opt):
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    spatial_transform = [
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor()
    ]
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        TemporalEvenCrop(opt.sample_duration, opt.n_val_samples))
    temporal_transform = TemporalCompose(temporal_transform)

    val_data, collate_fn = get_validation_data(opt.video_path, opt.annotation_path,
                                                opt.dataset,
                                               opt.input_type, opt.file_type,
                                               spatial_transform,
                                               temporal_transform)
    if opt.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data, shuffle=False)
    else:
        val_sampler = None
        
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=(opt.batch_size #//
                                                         #opt.n_val_samples
                                                         ),
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=collate_fn)

    if opt.is_master_node:
        val_logger = Logger(opt.result_path / 'val.log',
                            ['epoch', 'loss', 'acc'])
    else:
        val_logger = None

    return val_loader, val_logger


def get_inference_utils(opt):
    assert opt.inference_crop in ['center', 'nocrop']

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)

    spatial_transform = [Resize(opt.sample_size)]
    if opt.inference_crop == 'center':
        spatial_transform.append(CenterCrop(opt.sample_size))
    spatial_transform.append(ToTensor())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    # ------- edit here for other datasets
    temporal_transform = None
    #temporal_transform = []
    #if opt.sample_t_stride > 1:
    #    temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    # sliding window?????! we don't neet it when charades and holly2
    #temporal_transform.append(
    #    SlidingWindow(opt.sample_duration, opt.inference_stride))
    #temporal_transform = TemporalCompose(temporal_transform)
    #-----------------------------------------------
    
    # get here
    inference_data, collate_fn = get_inference_data(opt.video_path, opt.annotation_path, opt.dataset, opt.input_type,
        opt.file_type, opt.inference_subset, spatial_transform, temporal_transform)

    inference_loader = torch.utils.data.DataLoader(
        inference_data,
        batch_size=opt.inference_batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)

    return inference_loader, inference_data.class_names


def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)

"""
Main worker Starts here!
"""
def main_worker(index, opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    # logging
    loss_log = []
    acc_log = []
    val_acc_log = []
    #--

    if index >= 0 and opt.device.type == 'cuda':
        opt.device = torch.device(f'cuda:{index}')

    if opt.distributed:
        opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + index
        dist.init_process_group(backend='nccl',
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.dist_rank)
        opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
        opt.n_threads = int(
            (opt.n_threads + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    model = generate_model(opt)
    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if opt.pretrain_path:
        model = load_pretrained_model(model, opt.pretrain_path, opt.model,
                                      opt.n_finetune_classes)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)
    model = make_data_parallel(model, opt.distributed, opt.device)
    
    if opt.pretrain_path:
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)
    else:
        parameters = model.parameters()
    
    if opt.is_master_node:
        print(model)

    criterion = CrossEntropyLoss().to(opt.device)

    info = opt.dataset + "_" + opt.model + str(opt.model_depth) + '_M' + str(opt.M) + '_'    ###
    print(info)

    if not opt.no_train:
        (train_loader, train_sampler, train_logger, train_batch_logger,
         optimizer, scheduler) = get_train_utils(opt, parameters)
        if opt.resume_path is not None:
            opt.begin_epoch, optimizer, scheduler = resume_train_utils(
                opt.resume_path, opt.begin_epoch, optimizer, scheduler)
            if opt.overwrite_milestones:
                scheduler.milestones = opt.multistep_milestones
    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt)

    if opt.tensorboard and opt.is_master_node:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    prev_val_loss = None
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            if opt.distributed:
                train_sampler.set_epoch(i)
            current_lr = get_lr(optimizer)

            tmp_acc, tmp_loss = train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, current_lr, train_logger,
                        train_batch_logger, tb_writer, opt.distributed)


            acc_log.append(tmp_acc)
            loss_log.append(tmp_loss)

            print("Train Accuracy : ", tmp_acc, ", Loss : ", tmp_loss)

            if i % opt.checkpoint == 0 and opt.is_master_node:
                save_file_path = opt.result_path / '{}_save_{}.pth'.format(info, i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)

        if not opt.no_val:
            prev_val_loss, prev_val_acc = val_epoch(i, val_loader, model, criterion,
                                      opt.device, val_logger, tb_writer,
                                      opt.distributed)
            print("Validation Accuracy : ", prev_val_acc, ", Loss : ", prev_val_loss)
            val_acc_log.append(prev_val_acc)
            

        if not opt.no_train and opt.lr_scheduler == 'multistep':
            scheduler.step()
        elif not opt.no_train and opt.lr_scheduler == 'plateau':
            scheduler.step(prev_val_loss)

    if opt.inference:
        # inference here
        inference_loader, inference_class_names = get_inference_utils(opt)
        inference_result_path = opt.result_path / '{}.json'.format(
            opt.inference_subset)

        inference.inference(inference_loader, model, inference_result_path,
                            inference_class_names, opt.inference_no_average,
                            opt.output_topk)
    # save file
    # if attn not extract
    

    if not opt.inference:
        infoname = info + time.strftime("%Y%m%d-%H%M%S")

        timestr = "./results/" + infoname
        with open(timestr+'_val_acc.txt', 'w') as f:
            for i in range(len(val_acc_log)):
                f.write("%s " % round(val_acc_log[i],5))
        with open(timestr+'_train_acc.txt', 'w') as f:
            for i in range(len(acc_log)):
                f.write("%s " % round(acc_log[i],5))
        with open(timestr+'_loss.txt', 'w') as f:
            for i in range(len(loss_log)):
                f.write("%s " % round(loss_log[i],5))


if __name__ == '__main__':
    opt = get_opt()

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')

    opt.ngpus_per_node = torch.cuda.device_count()
    if opt.distributed:
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(main_worker, nprocs=opt.ngpus_per_node, args=(opt,))
    else:
        main_worker(-1, opt)