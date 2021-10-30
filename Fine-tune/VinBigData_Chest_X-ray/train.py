# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
from torchvision import models
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn import metrics

from sklearn.utils.multiclass import type_of_target
from scipy import ndimage
import pdb

logger = logging.getLogger(__name__)


CLASS_NAMES = ['Aortic enlargement',
 'Atelectasis',
 'Pneumothorax',
 'Lung Opacity',
 'Pleural thickening',
 'ILD',
 'Pulmonary fibrosis',
 'Calcification',
 'Pleural effusion',
 'Consolidation',
 'Cardiomegaly',
 'Other lesion',
 'Nodule-Mass',
 'Infiltration']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def auc(pred_property_array, one_hot_labels, num_classes):
    AUROCs = []
    # pdb.set_trace()
    for i in range(num_classes):
        AUROCs.append(roc_auc_score(one_hot_labels[:, i], pred_property_array[:, i]))
    # print(AUROCs)
    return AUROCs

def simple_accuracy(preds, labels):
    # print(preds)
    # print(labels)
    return ((preds == labels) * 1).mean()

def classification_report(preds, labels):
    return metrics.classification_report(labels,preds)

def save_model_auc(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_bestauc_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def save_model_loss(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_bestloss_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

# simclr
def load_weights(model, weight_path):
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}

    print("load weights")
    for k, _ in load_weights.items():
        print(k)

    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    return model


def setup(args):
    
    # Prepare model
    num_classes = args.num_classes
    # print(num_classes)
    if args.model_type == "ViT-B_16":
        config = CONFIGS[args.model_type]

        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
        model = load_weights(model, args.pretrained_dir)
        logger.info("{}".format(config))

    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    all_property = []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    # loss_fct = torch.nn.CrossEntropyLoss()
    loss_fct = torch.nn.BCEWithLogitsLoss()
    
    for step, batch in enumerate(epoch_iterator):
        # if step > 10:  # debug code 
        #     break
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            if  "net" in args.model_type:
                logits = model(x)#[0]
            else :
                logits = model(x)[0]
            eval_loss = loss_fct(logits, y.float())
            eval_losses.update(eval_loss.item())

            # preds = torch.argmax(logits, dim=-1)
            preds = (logits.sigmoid() > 0.5) * 1

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            all_property.append(logits.sigmoid().detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            all_property[0] = np.append(
                all_property[0], logits.sigmoid().detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label, all_property = all_preds[0], all_label[0], all_property[0]

    aurocs = auc(all_property, all_label, args.num_classes)
    auroc_avg = np.array(aurocs).mean()

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Auc: %2.5f" % auroc_avg)

    writer.add_scalar("valid/loss", scalar_value=eval_losses.avg, global_step=global_step)
    # writer.add_scalar("valid/auc", scalar_value=auroc_avg, global_step=global_step)
    return auroc_avg, eval_losses.avg

## 未对测试集做数据增强
def test(args, model, test_loader):
    # Test!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    all_property = []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    # loss_fct = torch.nn.CrossEntropyLoss()
    loss_fct = torch.nn.BCEWithLogitsLoss()
    
    for step, batch in enumerate(epoch_iterator):
        # if step > 10:  # debug code 
        #     break
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            if  "net" in args.model_type:
                logits = model(x)#[0]
            else :
                logits = model(x)[0]
            # print(logits.size())
            eval_loss = loss_fct(logits, y.float())
            eval_losses.update(eval_loss.item())

            # preds = torch.argmax(logits, dim=-1)
            preds = (logits.sigmoid() > 0.5) * 1
            
        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            all_property.append(logits.sigmoid().detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            all_property[0] = np.append(
                all_property[0], logits.sigmoid().detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label, all_property = all_preds[0], all_label[0], all_property[0]

    accuracy = simple_accuracy(all_preds, all_label)
    aurocs = auc(all_property, all_label, args.num_classes)
    auroc_avg = np.array(aurocs).mean()

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    print('The average AUROC is {auroc_avg:.3f}'.format(auroc_avg=auroc_avg))
    for i in range(14):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], aurocs[i]))

    path = "./output/"+ args.name+ "/" + args.pretrained_dir.split('/')[-1].split('.')[0]
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(len(CLASS_NAMES)):
        file_name = os.path.join(path,CLASS_NAMES[i] + '.txt')
        with open(file_name,"w",encoding="utf-8") as fw:
            my_label = all_label[:,i]
            my_all_property = all_property[:,i]
            for i in range(len(my_label)):
                fw.write(str(my_label[i]) + '\t' + str(my_all_property[i]) + '\n')

    with open(path+"/A_results.txt","w",encoding="utf-8") as fw:
        fw.write('AUC:' + '\t' + str(auroc_avg) + '\n')
        fw.write('\n')
        for i in range(len(CLASS_NAMES)):
            fw.write(CLASS_NAMES[i] + '\t' + str(aurocs[i]) + '\n')
        
def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))  #  tensorboard Supporting documents, in logs/name/

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    if args.model_type == "ViT-B_16":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_auc = 0, 0
    min_loss = 10000000
    last_auc = 10
    down = 0
    loss_fct = torch.nn.BCEWithLogitsLoss()
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # if step > 10:
            # break

            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            # pdb.set_trace()
            if  "net" in args.model_type:
                logits = model(x)#[0]
            else :
                logits = model(x)[0]
            loss = loss_fct(logits.view(-1, args.num_classes), y.float())
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                # if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                
                len_train = len(train_loader)
                if global_step % len_train == 0 and args.local_rank in [-1, 0]:
                    
                    auroc_avg, val_loss = valid(args, model, writer, test_loader, global_step)
                    if best_auc > auroc_avg:
                        down = down + 1
                    
                    else :
                        down = 0
                    print(down)

                    if best_auc < auroc_avg:
                        save_model_auc(args, model)
                        best_auc = auroc_avg

                    if min_loss > val_loss:
                        save_model_loss(args, model)
                        min_loss = val_loss
                if down > 10:
                    break
                if global_step % t_total == 0:
                    break
        losses.reset()
        if down > 10:
            break
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    
    logger.info("min_Loss: \t%f" % min_loss)
    # logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")

    parser.add_argument("--stage", type=str, default="train", help="train or test?")
    
    parser.add_argument("--model_type", choices=["ViT-B_16", "Resnet50", "Resnet18","Resnet101","Densenet121"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--num_classes",default = 14,type=int,help="the number of class")                    
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=384, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")               
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--data_volume", type=str)
    parser.add_argument("--gpu", type=str)

    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "32"
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        print('##############################')   
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                            timeout=timedelta(minutes=60)
                                            # init_method='tcp://127.0.0.1:23457',
                                            # world_size=2,
                                            # rank=args.local_rank,
                                            )
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    if args.stage == "train":
        # Training
        train(args, model)
    else :
        test_loader = get_loader(args)
        test(args, model, test_loader)


if __name__ == "__main__":
    main()
