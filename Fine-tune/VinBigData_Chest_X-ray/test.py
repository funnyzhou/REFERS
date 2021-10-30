import pdb
import os
filePath = "./output/"
filenames = os.listdir(filePath)
filenames.sort()

# parameter
gpu = "0"
name = "test1"

filenames = ["caption_100_bestauc_checkpoint.bin"]
for i in range(len(filenames)):
    print(filenames[i])
        # continue
    if  filenames[i].split('_')[0] == "caption":
        model = "ViT-B_16"
        # continue
    print(os.system('CUDA_VISIBLE_DEVICES=' + gpu +' python3 train.py --name ' + name + ' --stage test --model_type ' + model +' --num_classes 14 --pretrained_dir ' + filePath + filenames[i] +' --eval_batch_size 512 --img_size 224 --fp16 --fp16_opt_level O2'))
    # break
    

