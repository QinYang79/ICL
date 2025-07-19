import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import numpy as np
from matplotlib import pyplot as plt
from pylab import xticks,yticks,np
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture



def split_prob(prob, threshld):
    if prob.min() > threshld:
        """From https://github.com/XLearning-SCU/2021-NeurIPS-NCR"""
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print('No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.')
        threshld = np.sort(prob)[len(prob)//100]
    pred = (prob > threshld)
    return (pred+0)

def get_loss(model, data_loader):
    logger = logging.getLogger("RDE.train")
    model.eval()
    device = "cuda"
    data_size = data_loader.dataset.__len__()
    real_labels = data_loader.dataset.real_correspondences
    lossA, lossB, simsA,simsB = torch.zeros(data_size), torch.zeros(data_size), torch.zeros(data_size),torch.zeros(data_size)
    for i, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        index = batch['index']
        with torch.no_grad(): 
            la, lb, sa, sb = model.compute_per_loss(batch)
            for b in range(la.size(0)):
                lossA[index[b]]= la[b]
                lossB[index[b]]= lb[b]
                simsA[index[b]]= sa[b]
                simsB[index[b]]= sb[b]
            if i % 100 == 0:
                logger.info(f'compute loss batch {i}')

    losses_A = (lossA-lossA.min())/(lossA.max()-lossA.min())    
    losses_B = (lossB-lossB.min())/(lossB.max()-lossB.min())
    
    input_loss_A = losses_A.reshape(-1,1) 
    input_loss_B = losses_B.reshape(-1,1)
 
    logger.info('\nFitting GMM ...') 
 
    if model.args.noisy_rate > 0.4 or model.args.dataset_name=='RSTPReid':
        # should have a better fit 
        gmm_A = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
        gmm_B = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
    else:
        gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

    gmm_A.fit(input_loss_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(input_loss_A.cpu().numpy())
    prob_A = prob_A[:, gmm_A.means_.argmin()]

    gmm_B.fit(input_loss_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(input_loss_B.cpu().numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]
 
    pred_A = split_prob(prob_A, 0.5)
    pred_B = split_prob(prob_B, 0.5)
  
    return torch.Tensor(pred_A), torch.Tensor(pred_B)

def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("RDE.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "bge_loss": AverageMeter(),
        "tse_loss": AverageMeter(),
        "bge_aug_loss": AverageMeter(),
        "tse_aug_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    # evaluator.eval(model.eval())
    # train
    sims = []
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        model.epoch = epoch
        # data_size = train_loader.dataset.__len__()
        # pred_A, pred_B  =  torch.ones(data_size), torch.ones(data_size)
    
        pred_A, pred_B = get_loss(model, train_loader)
        consensus_division = pred_A + pred_B # 0,1,2 
        consensus_division[consensus_division==1] += torch.randint(0, 2, size=(((consensus_division==1)+0).sum(),))
        label_hat = consensus_division.clone()
        label_hat[consensus_division>1] = 1
        label_hat[consensus_division<=1] = 0 
        
        # model.train()
        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            index = batch['index']
            
            batch['label_hat'] = label_hat[index.cpu()]
 
            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['bge_loss'].update(ret.get('bge_loss', 0), batch_size)
            meters['tse_loss'].update(ret.get('tse_loss', 0), batch_size)
            meters['bge_aug_loss'].update(ret.get('bge_aug_loss', 0), batch_size)
            meters['tse_aug_loss'].update(ret.get('tse_aug_loss', 0), batch_size)
         
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
 
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1, _ = evaluator.eval(model.module.eval())
                else:
                    top1, _ = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
 
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")

    # arguments["epoch"] = epoch
    # checkpointer.save("last", **arguments)
                    
def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("RDE.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1, sims = evaluator.eval(model.eval())
    return sims
