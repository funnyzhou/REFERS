import pdb
import os
filePath = "./output/"
filenames = os.listdir(filePath)
filenames.sort()

# parameter
gpu = "7"
name = "test1"

filenames = ["caption_100_bestauc_checkpoint.bin"]
for i in range(len(filenames)):
    # pdb.set_trace()
    print(filenames[i])
    if filenames[i].split('_')[0] == "caption":
        model = "ViT-B_16"

    print(os.system('CUDA_VISIBLE_DEVICES=' + gpu +' python3 train.py --name ' + name + ' --stage test --model_type ' + model +' --num_classes 1 --pretrained_dir ' + filePath + filenames[i] +' --eval_batch_size 512 --img_size 224 --fp16 --fp16_opt_level O2'))
    # break
    

