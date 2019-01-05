import os
import sys
import glob
import time
import copy
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from model import NASNetwork

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'])
parser.add_argument('--data_path', type=str, default='/tmp/cifar10_data')
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=500)
parser.add_argument('--num_epochs', type=int, default=600)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--num_nodes', type=int, default=5)
parser.add_argument('--out_filters', type=int, default=36)
parser.add_argument('--lr_T_0', type=int, default=None)
parser.add_argument('--lr_T_mul', type=int, default=None)
parser.add_argument('--cutout_size', type=int, default=None)
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--lr_max', type=float, default=None)
parser.add_argument('--lr_min', type=float, default=None)
parser.add_argument('--keep_prob', type=float, default=0.5)
parser.add_argument('--drop_path_keep_prob', type=float, default=1.0)
parser.add_argument('--l2_reg', type=float, default=1e-4)
parser.add_argument('--fixed_arc', type=str, default=None)
parser.add_argument('--use_aux_head', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

def train(train_queue, model, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(async=True)
    
        optimizer.zero_grad()
        logits, aux_logits = model(input)
        loss = model.loss(logits, target)
        if not aux_logits:
            aux_loss = model.loss(aux_logits, target)
            loss += 0.4 * aux_loss
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_bound)
        optimizer.step()
    
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)
    
        if step % 100 == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def valid(valid_queue, model):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)
    
        logits, _ = model(input)
        loss = model.loss(logits, target)
    
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)
    
        if step % 100 == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU found!')
        sys.exit(1)
    
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manmual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    
    args.num_steps = np.ceil(args.num_epochs * 50000 / args.batch_size)
    
    logging.info("Args = %s", args)
    
    model = NASNetwork(args.num_layers, args.num_nodes, args.out_filters, args.keep_prob, args.drop_path_keep_prob, args.use_aux_head, args.num_steps, args.arch)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr_max,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.num_epochs))
    
    _, model_state_dict, epoch, scheduler_state_dict, optimizer_state_dict = utils.load(args.output_dir)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    
    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        train_acc, train_obj = train(train_queue, model, optimizer)
        logging.info('train_acc %f', train_acc)
        valid_acc, valid_obj = valid(valid_queue, model)
        logging.info('valid_acc %f', valid_acc)
        utils.save(args.output_dir, args, model, epoch, scheduler, optimizer)
        

if __name__ == '__main__':
    main()