import torch
from model import resnet34
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
import torch.utils.data as data
import numpy as np 
import pandas as pd 
from PIL import Image
import cv2
import logging
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

logger = logging.getLogger(__name__)




def simple_accuracy(preds, labels):
    # print(preds)
    # print(labels)
    return (preds == labels).mean()




def load_weights(model, weight_path):
    pretrained_weights = torch.load(weight_path)
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}
    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    return model


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    # num_classes =  if args.dataset == "cifar10" else 100
    num_classes = args.num_classes
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    # model.load_from(np.load(args.pretrained_dir))
    model = load_weights(model, args.pretrained_dir)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000



class XRAY(data.Dataset):
    
    train_data = "train_data.txt"
    val_data = "val_data.txt"
    test_data = "test_data.txt"
    
    train_label = "train_label.npy"
    val_label = "val_label.npy"
    test_label = "test_label.npy"    
    
    
    def __init__(self, root, split="train", transform=None):
        super(XRAY, self)
        self.split = split
        self.root = root
        self.transform = transform
        self.data = []# 装图片路径
#         self.targets = [] # 装图片标签
        
        if self.split == "train":
            # 训练集的路径
            downloaded_data_txt = self.train_data
            downloaded_label_txt= self.train_label
        
        elif self.split == "val":
            downloaded_data_txt = self.val_data
            downloaded_label_txt= self.val_label      
            
        elif self.split == "test":
            downloaded_data_txt = self.test_data
            downloaded_label_txt= self.test_label 
        
        
        with open(os.path.join(self.root,downloaded_data_txt),"r",encoding="utf-8") as fr:
            data_list = fr.readlines()
        for i in range(len(data_list)):
            if data_list[i][-1] == '\n':
                self.data.append(data_list[i][:-1])
            else :
                self.data.append(data_list[i])
        self.targets = np.load(os.path.join(root,downloaded_label_txt))
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_test_loader():


    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    trainset = XRAY(root = "./DataProcessed/VinBigData_Chest_X-ray/", split="train", transform= transform_train)

    testset = XRAY(root = "./DataProcessed/VinBigData_Chest_X-ray/", split="val", transform= transform_test)


    test_sampler = SequentialSampler(testset)

    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=64,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return test_loader


def valid(args, model, test_loader):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    # loss_fct = torch.nn.CrossEntropyLoss()
    loss_fct = torch.nn.BCEWithLogitsLoss()
    sigmoid = torch.nn.Sigmoid() 
    
    for step, batch in enumerate(epoch_iterator):
        # if step > 10:  # debug code 
        #     break
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]
            eval_loss = loss_fct(logits, y.float())
            eval_losses.update(eval_loss.item())

            # preds = torch.argmax(logits, dim=-1)
            preds = (logits.sigmoid() > 0.5) * 1

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    return accuracy, eval_losses.avg


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")
    
    parser.add_argument("--num_classes",default = 15,type=int,help="the number of class")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
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

    os.environ["CUDA_VISIBLE_DEVICES"] = '7, 8'

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
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

    # Testing
	dataloaders = get_test_loader()
    valid(args, model, dataloaders)


if __name__ == "__main__":
    main()


