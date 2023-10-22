# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import gc
from tqdm import tqdm
import collections.abc as container_abcs

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import torch.nn as nn
import torchvision
from collections import OrderedDict

from datasets import build_dataset
from engine import train_one_epoch, evaluate
# import models
# import utils

def get_args_parser():
    parser = argparse.ArgumentParser('Conviformer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=1, type=int)

    # Model parameters
    parser.add_argument('--model', default='densenet', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--pretrained', action='store_true')

    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--embed_dim', default=48, type=int, help='embedding dimension per head')

     # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')


    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--sampling_ratio', default=1.,
                        type=float, help='fraction of samples to keep in the training set of imagenet')
    parser.add_argument('--nb_classes', default=None,
                        type=int, help='number of classes in imagenet')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)





    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    return parser

def main(args):


    print(args)

    device = torch.device("cpu")
    
    # fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    
    dataset_train, args.nb_classes = build_dataset(which_data='train', args=args)
    dataset_val, _ = build_dataset(which_data='val', args=args)[:2]
                

    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    sampler_val = torch.utils.data.RandomSampler(
                dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        pin_memory=args.pin_mem, drop_last=False
    )

    cls_num_list = torch.zeros(args.nb_classes, dtype=torch.long)

   
    print(f"Creating model: {args.model}")

    model = torchvision.models.densenet169(pretrained=True)

    print(model)

    for param in model.parameters():
        param.requires_grad = False
    n_inputs = model.classifier.in_features
    last_layer = nn.Linear(n_inputs, args.nb_classes)
    model.classifier = last_layer

    
    
    

    linear_scaled_lr = args.lr * args.batch_size # * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = torch.optim.Adam(model.classifier.parameters())

    loss_scaler = scaler = torch.cuda.amp.GradScaler(enabled=True)

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = torch.nn.CrossEntropyLoss()
    
    output_dir = Path(args.output_dir)
    torch.save(args, output_dir / "args.pyT")


    # if args.eval:
    #     throughput = utils.compute_throughput(model, resolution=args.input_size)
    #     print(f"Throughput : {throughput:.2f}")
    #     test_stats = evaluate(data_loader_val, model, device)
    #     print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    #     return


    print("Start training")
    start_time = time.time()
    max_accuracy = 0.0

    training_history = {'accuracy':[],'loss':[]}
    validation_history = {'accuracy':[],'loss':[]}

    from tqdm import tqdm

    def train(trainloader, model, criterion, optimizer, scaler, device=torch.device("cpu")):
        train_acc = 0.0
        train_loss = 0.0
        for images, labels in tqdm(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                output = model(images)
                loss = criterion(output, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                acc = ((output.argmax(dim=1) == labels).float().mean())
                train_acc += acc
                train_loss += loss
        return train_acc/len(trainloader), train_loss/len(trainloader)



    epochnum=[]
    train_accuracy=[]
    train_losses=[]
    evaluate_accuracy=[]
    evaluate_loss=[]
    max_score = 0

    for epoch in range(args.epochs):
        train_acc, train_loss = train(data_loader_train, model, criterion, optimizer, scaler, device=device)
        # eval_acc, eval_loss = evaluate(val_loader, model, criterion, device=device)
        epochnum.append(epoch)
        train_accuracy.append(train_acc)
        train_losses.append(train_loss)
        # evaluate_accuracy.append(eval_acc)
        # evaluate_loss.append(eval_loss)
        
        print("")
        print(f"Epoch {epoch + 1} | Train Acc: {train_acc*100} | Train Loss: {train_loss}")
        # print(f"\t Val Acc: {eval_acc*100} | Val Loss: {eval_loss}")
        print("===="*8)
        torch.save(model, './all_model.pth')

            

        # lr_scheduler.step(epoch)
        # # if args.output_dir:
        # #     checkpoint_paths = [output_dir / 'checkpoint.pth']
        # #     if args.save_every is not None:
        # #         if epoch % args.save_every == 0: checkpoint_paths.append(output_dir / 'checkpoint_{}.pth'.format(epoch))
        # #     for checkpoint_path in checkpoint_paths:
        # #         utils.save_on_master({
        # #             'model': model_without_ddp.state_dict(),
        # #             'optimizer': optimizer.state_dict(),
        # #             'lr_scheduler': lr_scheduler.state_dict(),
        # #             'epoch': epoch,
        # #             'model_ema': get_state_dict(model_ema) if model_ema else None,
        # #             'args': args,
        # #         }, checkpoint_path)

        # test_stats = evaluate(data_loader_val, model, device)
        # if max_accuracy <= test_stats["acc1"]:
        #     max_accuracy = test_stats["acc1"]
        #             # if args.output_dir:
        #             #     checkpoint_paths = [output_dir / 'best-checkpoint.pth']
        #             #     utils.save_on_master({
        #             #         'model': model_without_ddp.state_dict(),
        #             #         'optimizer': optimizer.state_dict(),
        #             #         'lr_scheduler': lr_scheduler.state_dict(),
        #             #         'epoch': epoch,
        #             #         'model_ema': get_state_dict(model_ema) if model_ema else None,
        #             #         'args': args,
        #             #     }, checkpoint_path)
                        
        # print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        # max_accuracy = max(max_accuracy, test_stats["acc1"])
        # print(f'Max accuracy: {max_accuracy:.2f}%')

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              **{f'test_{k}': v for k, v in test_stats.items()},
        #              'epoch': epoch}
        # print(log_stats)

        # if args.output_dir: # and utils.is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
