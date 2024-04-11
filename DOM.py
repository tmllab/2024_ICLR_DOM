import argparse
import logging
import sys
import time
import math
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

from wideresnet import WideResNet
from preactresnet import PreActResNet18
from utils import *

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()


def normalize(X):
    return (X - mu)/std

upper_limit, lower_limit = 1,0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, norm):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        index = slice(None,None,None)
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def data_augment(model, X, y, tran_x, clamp, process_transform, max_arg_iteration, arg_strength):
    count_i = 1
    while count_i <= max_arg_iteration:
        X_index = torch.where(tran_x)[0]
        X_arg= torch.clamp(process_transform(X[X_index].clone()), min=lower_limit, max=upper_limit)
        output = model(normalize(X_arg))
        loss = nn.CrossEntropyLoss(reduce=False)(output, y[X_index])
        Change_X = (loss >= clamp)
        Change_index = torch.where(Change_X)[0]
        Final_index = X_index[Change_index]
        if count_i == max_arg_iteration:
            X[X_index] = X_arg
        else:
            X[X_index] = X[X_index] * (1 - arg_strength) + X_arg * arg_strength
            X[Final_index] = X_arg[Change_index]
        tran_x[Final_index] = False
        count_i += 1
        if tran_x.sum().item() == 0:
            break
    return X

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--batch-size-test', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='piecewise')
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--attack', default='pgd', type=str, choices=['fgsm','pgd', 'none'])
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--epsilon', default=8, type=float)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--pgd-step-train', default=10, type=int)
    parser.add_argument('--pgd-step-test', default=20, type=int)
    parser.add_argument('--fgsm-alpha', default=10, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fname', default='cifar_model_baseline_AT', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--chkpt-iters', default=1000, type=int)
    parser.add_argument('--clamp', default=1.0, type=float)
    parser.add_argument('--operate', default='RE', type=str, choices=['RE', 'DA_AUG', 'DA_Rand'])
    parser.add_argument('--max-arg-iteration', default=0, type=int)
    parser.add_argument('--arg-strength', default=0.0, type=float)

    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = [Crop(32, 32), FlipLR()]

    dataset = cifar10(args.data_dir)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.), dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size_test, shuffle=False, num_workers=2)

    pgd_alpha = (args.pgd_alpha / 255.)
    fgsm_alpha = (args.fgsm_alpha / 255.)
    epsilon = (args.epsilon / 255.)

    if args.model == 'PreActResNet18':
        model = PreActResNet18(num_classes=10)
    elif args.model == 'WideResNet':
        model = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).cuda()
    params = model.parameters()
    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    epochs = args.epochs

    def lr_schedule(t, epochs):

        if args.attack == 'fgsm':
            if t <= (epochs/2):
                return args.lr_max * t / (epochs/2)
            elif t > (epochs/2):
                return args.lr_max * (epochs - t) / (epochs/2)
        else:
            if t <= epochs / 2:
                return args.lr_max
            elif t <= epochs * 3 / 4:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.

    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')
    else:
        start_epoch = 0

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    best_test_robust_acc = 0
    logger.info('V1.0.3')
    if args.attack == 'fgsm':
        logger.info('Epoch \t Train Time \t\t Test Time \t LR \t\t Train NA Loss \t Train NA Acc \t Train FGSM Loss \t Train FGSM Acc \t Train RO Loss \t Train RO Acc  \t Test NA Loss \t Test NA Acc  \t Test FGSM Loss \t Test FGSM Acc \t Test RO Loss \t Test RO Acc')
    elif args.attack == 'pgd':
        logger.info('Epoch \t Train Time \t\t Test Time \t LR \t\t Train NA Loss \t Train NA Acc \t Train RO Loss \t Train RO Acc  \t Test NA Loss \t Test NA Acc  \t Test RO Loss \t Test RO Acc')
    elif args.attack == 'none':
        logger.info('Epoch \t Train Time \t\t Test Time \t LR \t\t Train NA Loss \t Train NA Acc \t Test NA Loss \t Test NA Acc')
    for epoch in range(start_epoch, epochs):
        start_time = time.time()

        for i, batch in enumerate(train_batches):
            if args.eval:
                break
            X, y = batch['input'], batch['target']
            training_t = epoch + ((i + 1) / len(train_batches))
            lr = lr_schedule(training_t, epochs)
            opt.param_groups[0].update(lr=lr)

            output = model(normalize(X))
            loss = nn.CrossEntropyLoss(reduce=False)(output, y)

            loss_clamp = (loss >= args.clamp).detach()

            if args.operate != 'RE' and training_t >= (epochs/2):
                if args.operate == 'DA_AUG':
                    process_transform = torchvision.transforms.Compose([
                    torchvision.transforms.ConvertImageDtype(torch.uint8),
                    torchvision.transforms.AugMix(),
                    torchvision.transforms.ConvertImageDtype(torch.float)
                    ])
                elif args.operate == 'DA_Rand':
                    process_transform = torchvision.transforms.Compose([
                    torchvision.transforms.ConvertImageDtype(torch.uint8),
                    torchvision.transforms.RandAugment(),
                    torchvision.transforms.ConvertImageDtype(torch.float)
                    ])

                tran_x = (loss <= args.clamp).detach()
                if tran_x.sum().item() != 0:
                    X_temp = X.clone().detach()
                    X_afterArg = data_augment(model, X_temp, y, tran_x, args.clamp, process_transform, args.max_arg_iteration, args.arg_strength)
                    X = X_afterArg

            if args.attack == 'pgd':
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.pgd_step_train, args.restarts, args.norm)
                delta = delta.detach()
            elif args.attack == 'fgsm':
                delta = attack_pgd(model, X, y, epsilon, fgsm_alpha , 1, args.restarts, args.norm)
                delta = delta.detach()
            elif args.attack == 'none':
                delta = torch.zeros_like(X)
                delta = delta.detach()
            X_adv = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))

            model.train()
            robust_output = model(X_adv)
            robust_loss = nn.CrossEntropyLoss(reduce=False)(robust_output, y)

            if args.operate == 'RE' and training_t >= (epochs/2):
                robust_loss = robust_loss * loss_clamp
            robust_loss = robust_loss.mean()

            opt.zero_grad()
            robust_loss.backward()
            opt.step()

        train_time = time.time()

        model.eval()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_loss_fgsm = 0
        train_acc_fgsm = 0
        train_n = 0.0
        for i, batch in enumerate(train_batches):
            X, y = batch['input'], batch['target']
            if args.attack == 'none':
                delta = torch.zeros_like(X)
            elif args.attack == 'pgd':
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.pgd_step_train, args.restarts, args.norm)
            elif args.attack == 'fgsm':
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.pgd_step_train, args.restarts, args.norm)
                delta_fgsm = attack_pgd(model, X, y, epsilon, fgsm_alpha, 1, args.restarts, args.norm)
            delta = delta.detach()

            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            robust_loss = criterion(robust_output, y)

            output = model(normalize(X))
            loss = criterion(output, y)

            if args.attack == 'fgsm':
                delta_fgsm = delta_fgsm.detach()
                robust_output_fgsm = model(normalize(torch.clamp(X + delta_fgsm[:X.size(0)], min=lower_limit, max=upper_limit)))
                robust_loss_fgsm = criterion(robust_output_fgsm, y)
                train_loss_fgsm += robust_loss_fgsm.item() * y.size(0)
                train_acc_fgsm += (robust_output_fgsm.max(1)[1] == y).sum().item()

            train_robust_loss += robust_loss.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        test_loss_0 = 0
        test_acc_0 = 0
        test_loss_8 = 0
        test_acc_8 = 0
        test_loss_8_fgsm = 0
        test_acc_8_fgsm = 0
        test_n = 0.0
        for i, batch in enumerate(test_batches):
            X, y = batch['input'], batch['target']
            if args.attack == 'none':
                delta_8 = torch.zeros_like(X)
            elif args.attack == 'pgd':
                delta_8 = attack_pgd(model, X, y, epsilon, pgd_alpha, args.pgd_step_test, args.restarts, args.norm)
            elif args.attack == 'fgsm':
                delta_8 = attack_pgd(model, X, y, epsilon, pgd_alpha, args.pgd_step_test, args.restarts, args.norm)
                delta_8_fgsm = attack_pgd(model, X, y, epsilon, fgsm_alpha, 1, args.restarts, args.norm)

            delta_8 = delta_8.detach()
            robust_output_8 = model(normalize(torch.clamp(X + delta_8[:X.size(0)], min=lower_limit, max=upper_limit)))
            robust_loss_8 = criterion(robust_output_8, y)

            output_0 = model(normalize(X))
            loss_0 = criterion(output_0, y)

            if args.attack == 'fgsm':
                delta_8_fgsm = delta_8_fgsm.detach()
                robust_output_8_fgsm = model(normalize(torch.clamp(X + delta_8_fgsm[:X.size(0)], min=lower_limit, max=upper_limit)))
                robust_loss_8_fgsm = criterion(robust_output_8_fgsm, y)
                test_loss_8_fgsm += robust_loss_8_fgsm.item() * y.size(0)
                test_acc_8_fgsm += (robust_output_8_fgsm.max(1)[1] == y).sum().item()

            test_loss_8 += robust_loss_8.item() * y.size(0)
            test_acc_8 += (robust_output_8.max(1)[1] == y).sum().item()
            test_loss_0 += loss_0.item() * y.size(0)
            test_acc_0 += (output_0.max(1)[1] == y).sum().item()
            test_n += y.size(0)

        test_time = time.time()

        if not args.eval:
            if args.attack == 'fgsm':
                logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                    epoch, train_time - start_time, test_time - train_time, lr,
                    train_loss/train_n, train_acc/train_n, train_loss_fgsm/train_n, train_acc_fgsm/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
                    test_loss_0/test_n, test_acc_0/test_n, test_loss_8_fgsm/test_n, test_acc_8_fgsm/test_n, test_loss_8/test_n, test_acc_8/test_n
                )
            elif args.attack == 'pgd':
                logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                    epoch, train_time - start_time, test_time - train_time, lr,
                    train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
                    test_loss_0/test_n, test_acc_0/test_n, test_loss_8/test_n, test_acc_8/test_n
                )
            elif args.attack == 'none':
                logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                    epoch, train_time - start_time, test_time - train_time, lr, train_loss/train_n, train_acc/train_n, test_loss_0/test_n, test_acc_0/test_n
                )

            # save checkpoint
            if test_acc_8/test_n > best_test_robust_acc:
                torch.save(model.state_dict(), os.path.join(args.fname, f'model_best.pth'))
                best_test_robust_acc = test_acc_8/test_n
            if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
                torch.save(model.state_dict(), os.path.join(args.fname, f'model_{epoch}.pth'))

        else:
            return

if __name__ == "__main__":
    main()
