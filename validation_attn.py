import torch
import time
import sys

import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy


import matplotlib.pyplot as plt
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
    plt.savefig("result_attns/"+title+"Attn.png")
    
    # for one data
    bat = 15
    fig, axs = plt.subplots(B//4 + B%4 ,4,figsize=(B*2,B))
    for b in range(B):
        axs[b//4, b%4].set_title(video_ids[bat]+"Attention in Block" + str(b) + " M" + str(M))
        for m in range(M):
            y = [m for i in range(len(attns[b][m][bat]))]
            axs[b//4, b%4].scatter(attns[b][m][bat], y, c=colors[m], alpha=0.1)
    plt.savefig("result_attns/"+title+"One_Attn.png")
#        plt.savefig("Attn_b"+str(b)+"_batch"+str(batch)+".png")

def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              logger,
              tb_writer=None,
              distributed=False):
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
            
            targets = targets.to(device, non_blocking=True)
            outputs, attns = model(inputs)
            print(len(attns))
            
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
            plot_attns(attns, video_ids)
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
