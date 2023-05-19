from operator import mod
import torch
import random
import numpy as np
import argparse
import os
import sys
import math
import time
import shutil
import pickle
import torch.nn as nn
from classifier.vcop_submit import CoOp
import dataset.incremental_dataloader as incremental_dataloader
from utils import mkdir_p


def parse_option():
    parser = argparse.ArgumentParser('Prompt Learning for CLIP', add_help=False)

    parser.add_argument("--root", type=str, default='data_path',help='root')
    parser.add_argument("--aug",type=str, default='flip', help='root')

    parser.add_argument("--mean_per_class", action='store_true', help='mean_per_class')
    parser.add_argument("--db_name", type=str, default='imagenet100', help='dataset name')
    parser.add_argument("--num_runs", type=int, default=1, help='num_runs')
    parser.add_argument("--seed", type=int, default=0, help='random seed')

    parser.add_argument("--arch", type=str, default='ViT-L-16', help='arch')
    parser.add_argument("--checkpoint", type=str, default='path', help='save_checkpoint')
    parser.add_argument("--ckpt_path", type=str, default=None, help='ckpt_path')
    parser.add_argument("--keyprompt_path", type=str, default='path', help='keyprompt_path')
    parser.add_argument("--save_path", type=str, default='path', help='save_path')
    parser.add_argument("--ID", type=str, default='description', help='description')

    # optimization setting
    parser.add_argument("--lr", type=float, default=1e-3, help='num_runs')
    parser.add_argument("--wd", type=float, default=0.0, help='num_runs')
    parser.add_argument("--epochs", type=int, default=10, help='num_runs')
    parser.add_argument("--train_batch", type=int, default=32, help='num_runs')
    parser.add_argument("--test_batch", type=int, default=32, help='num_runs')

    #model setting
    parser.add_argument("--model", type=str, default='coop', help='model')
    parser.add_argument("--n_prompt", type=int, default=32, help='num_runs')
    parser.add_argument("--prompt_bsz", type=int, default=4, help='num_runs')

    #incremental setting
    parser.add_argument("--num_class", type=int, default=100, help='num_class')
    parser.add_argument("--class_per_task", type=int, default=10, help='class per task')
    parser.add_argument("--num_task", type=int, default=10, help='num_task')
    parser.add_argument("--start_sess", type=int, default=0, help='start session')
    parser.add_argument("--sess", type=int, default=0, help='current session')
    parser.add_argument("--memory", type=int, default=0, help='memory')
    parser.add_argument("--num_test", type=int, default=15, help='num_test_text')
    parser.add_argument("--num_prompt", type=int, default=10, help='num_prompt')
    parser.add_argument("--text_prompt", type=int, default=3, help='text_prompt')
    parser.add_argument("--keep", type=bool, default=False, help='keep')#continue from other datasets



    args, unparsed = parser.parse_known_args()

    args.mean_per_class = False
    if args.db_name in ['oxfordpets', 'fgvc_aircraft', 'oxford_flowers', 'caltech101']:
        args.mean_per_class = True

    if args.db_name in ['cifar10', 'eurosat','stl1e']:
        args.num_runs= 10
    # args.num_runs = 1

    if args.ckpt_path is None:
        args.ckpt_path = 'path/{}.pt'.format(args.arch)
    
    args.save_path = args.save_path + '/' + args.db_name + '/' + args.ID


    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    prev_key, prev_prompt = False, False
    setup_seed(args.seed)
    if args.keep==True:
        path_key=os.path.join(args.keyprompt_path, 'text_key.pth.tar')
        path_prompt=os.path.join(args.keyprompt_path, 'text_prompt.pth.tar') 
        prev_key=torch.load(path_key)
        prev_prompt=torch.load(path_prompt)
        print('prompt trained from previous dataset')
    else:
        print('prompt trained from random')
    if args.model == 'coop':
        model = CoOp(prev_key,prev_prompt,args=args,keep=args.keep)

    if not os.path.isdir(args.ckpt_path):
        mkdir_p(args.checkpoint)
    if not os.path.isdir(args.save_path):
        mkdir_p(args.save_path)
    np.save(args.checkpoint + "/seed.npy", args.seed)
    try:
        shutil.copy2('main_incremental_submit.py', args.checkpoint)
        shutil.copy2('vcop_submit.py', args.checkpoint)
    except:
        pass
    inc_dataset = incremental_dataloader.IncrementalDataset(
                        dataset_name=args.db_name,
                        args = args,
                        random_order=False, #random class
                        shuffle=True,
                        seed=args.seed,
                        batch_size=args.train_batch,
                        workers=8,
                        validation_split=0,
                        increment=args.class_per_task,
                    )
        
    start_sess = args.start_sess
    memory = None
    
    for ses in range(start_sess, args.num_task):
        task_info, train_loader, class_name, test_class, val_loader, test_loader, for_memory = inc_dataset.new_task(memory) 
        
        args.sess=ses         
        if(start_sess==ses and start_sess!=0): 
            inc_dataset._current_task = ses
            with open(args.save_path + "/sample_per_task_testing_"+str(args.sess-1)+".pickle", 'rb') as handle:
                sample_per_task_testing = pickle.load(handle)
            inc_dataset.sample_per_task_testing = sample_per_task_testing
            args.sample_per_task_testing = sample_per_task_testing             
            
        print('ses:',ses)
        print(task_info)    
        print(inc_dataset.sample_per_task_testing)     # dict{task:len(test)}
        args.sample_per_task_testing = inc_dataset.sample_per_task_testing
        len_train = task_info['n_train_data']
        

        data = {'train_loader': train_loader, 'class_names': class_name}
        model.fit(data,len_train)
        print('finish fit')
        torch.save(model.model.state_dict()['text_key'], os.path.join(args.save_path, 'text_key.pth.tar'))
        torch.save(model.model.prompt_learner.state_dict()['text_prompt'], os.path.join(args.save_path, 'text_prompt.pth.tar'))
        acc = model.accuracy(test_loader, args.num_test, test_class, mean_per_class=args.mean_per_class)
        print('acc',acc)

        with open(args.save_path + "/memory_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args.save_path + "/acc_task_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(acc, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(args.save_path + "/sample_per_task_testing_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(args.sample_per_task_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

if __name__ == '__main__':
    args = parse_option()
    main(args)