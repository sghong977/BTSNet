import torch
import time
import sys
import torchvision

import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy


import matplotlib.pyplot as plt
import numpy as np

path = "result_attns/"
#i_batch = 19                     # print index


# THIS IS TEMPORAL & CHANNEL ATTENTION (if visualize)
# attn here
# attns [# of SKConv][batch_size][M][channels][TEMPORAL]
def plot_attns_temporal(attns):
    global i_batch
    print(len(attns), len(attns[0]), len(attns[0][0]), len(attns[0][0][0]), len(attns[0][0][0][0]))
    title = "UCF_BTS_M4_ep200_temporal"
    B = len(attns)              # blocks
    M = len(attns[0][0])
    batch = 1                    #len(attns[0])
    pr = 3                 # per_row. plots per row

    colors = ['red',  'cyan', 'blue', 'magenta']
    fig, axs = plt.subplots(B//pr + B%pr ,pr,figsize=(B*1.5,B))    
    """
    for b in range(B):
        for m in range(M):
            for bat in range(batch):
                for tp in range(len(attns[b][bat][m][0])):
                    tmp = np.asarray(attns[b][bat][m])
                    y = [tp for i in range(len(tmp))]
                    tmp = np.transpose(tmp)    # [TEMPORAL][CHANNEL]

                    axs[b//pr, b%pr].set_title("Attention in Block" + str(b))
                    axs[b//pr, b%pr].scatter(tmp[tp], y, c=colors[m], alpha=0.3)
                    print(tp)
            print("m", m)
    plt.savefig(path+title+"Attn.png")
    """
    # averaging channels
    fig, axs = plt.subplots(B//pr + B%pr ,pr,figsize=(B*1.8,B*1.1))    
    for b in range(B):
        for m in range(M):
            tmp = np.asarray(attns[b][i_batch][m])
            tmp = (tmp - 0.25) * 1000        #
            tmp = np.transpose(tmp)    # [TEMPORAL][CHANNEL]
            tmp = np.average(tmp, axis=1)                    # channel average

            axs[b//pr, b%pr].set_title("Block" + str(b))
            axs[b//pr, b%pr].plot(tmp, c=colors[m])
    plt.savefig(path+title+str(i_batch)+"_Attn_avg.png")

    fig, axs = plt.subplots(B//pr + B%pr ,pr,figsize=(B*1.5,B*1.1))    
    for b in range(B):
        for m in range(M):
            for tp in range(len(attns[b][i_batch][m][0])):
                tmp = np.asarray(attns[b][i_batch][m])
                tmp = (tmp)# - 0.25)       #
                y = [tp for i in range(len(tmp))]
                tmp = np.transpose(tmp)    # [TEMPORAL][CHANNEL]
                axs[b//pr, b%pr].set_title("Block" + str(b))
                axs[b//pr, b%pr].scatter(y, tmp[tp], c=colors[m], alpha=0.5)
    plt.savefig(path+title+str(i_batch)+"_Attn.png")



#if visualize
# attn here
# attns [# of SKConv][M][batch_size][channels]
def plot_attns(attns):
    title = "Jester_M4_ep100"
    B = len(attns)
    M = len(attns[0])
    batch = len(attns[0][0])
    colors = ['red',  'blue', 'purple', 'green']
    fig, axs = plt.subplots(B//4 + B%4 ,4,figsize=(B*2,B))    
    for b in range(B):
        for m in range(M):
            for bat in range(batch):
                axs[b//4, b%4].set_title("Attention in Block" + str(b) + " M" + str(M))
                y = [m for i in range(len(attns[b][m][bat]))]
                axs[b//4, b%4].scatter(attns[b][m][bat], y, c=colors[m], alpha=0.1)
    plt.savefig(path+title+"Attn.png")
    
    # for one data
    bat = 15
    fig, axs = plt.subplots(B//4 + B%4 ,4,figsize=(B*2,B))
    for b in range(B):
        axs[b//4, b%4].set_title(video_ids[bat]+"Attention in Block" + str(b) + " M" + str(M))
        for m in range(M):
            y = [m for i in range(len(attns[b][m][bat]))]
            axs[b//4, b%4].scatter(attns[b][m][bat], y, c=colors[m], alpha=0.1)
    plt.savefig(path+title+"One_Attn.png")
#        plt.savefig("Attn_b"+str(b)+"_batch"+str(batch)+".png")

def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              logger,
              tb_writer=None,
              distributed=False):
    global i_batch
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            inputs = inputs.cuda()     # remove this!
            # get name
            #video_ids, segments = zip(*targets)
            #---
            _inputs = torch.transpose(inputs, 2,1)
            #_inputs = torch.transpose(_inputs, 2, -1)
            #[batch][temp][w,h][channel]
            _inputs = _inputs.detach().cpu()
            
            #---
            
            targets = targets.to(device, non_blocking=True)
            outputs, attns = model(inputs, attn=True)
            
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))

            # if visualize, break here.
            #plot_attns(attns, video_ids)
            for i_batch in range(len(inputs)):
                grd = torchvision.utils.make_grid(_inputs[i_batch])
                torchvision.utils.save_image(grd, path+str(i_batch)+"_batch.png")

                plot_attns_temporal(attns)
            break


    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    if logger is not None:
        logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/acc', accuracies.avg, epoch)

    return losses.avg, accuracies.avg # accuracies.avg added
