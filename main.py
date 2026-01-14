import datetime
import os
import random
import time

import numpy as np
import torch
from Args import args_parser
from Data import Data
from Node import Global_Node, Node
from Trainer import Trainer
from utils import Recorder, Summary, LR_scheduler, fix_seed


# init args
args = args_parser()
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
args.split = args.node_num
args.global_model = args.local_model

# fix seed
fix_seed(args.seed)

# torch.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)


print('Running on', args.device)
Data = Data(args)
Train = Trainer(args)
recorder = Recorder(args)

# logs
Summary(args)


current_time = datetime.datetime.now()


# init nodes
Global_node = Global_Node(Data.test_loader, args)
Edge_nodes = [Node(k, Data.train_loader[k], Data.test_loader, args) for k in range(args.node_num)]

Global_node.unlearning_data = Data.unlearning_loader
Global_node.retraining_data = Data.retraining_loader
Global_node.unlearning_data_eval = Data.unlearning_loader_eval
Global_node.retraining_data_eval = Data.retraining_loader_eval
Edge_nodes[args.misclient].unlearning_data = Data.unlearning_loader
Edge_nodes[args.misclient].retraining_data = Data.retraining_loader


startt = time.time()
spend_time = 0.0
# train
for rounds in range(args.R):
    roundtime1 = time.time()
    print('===============The {:d}-th round==============='.format(rounds + 1))
    if args.lr_sch:
        if rounds in [50, 80]:
            args.lr *= 0.5
        LR_scheduler(rounds, Edge_nodes, args)

    for k in range(len(Edge_nodes)):
        Edge_nodes[k].fork(Global_node)        
        for epoch in range(args.E):
            Train(Edge_nodes[k], rounds)

        print('--------------------------------')
    print('----------------Validation----------------')
    Global_node.merge(Edge_nodes)    
    roundtime = time.time() - roundtime1
    spend_time += roundtime 
    print(f"--- Round Time on {args.dataset}:    {roundtime:.4f}s")

    # log
    metrics_dict = recorder.validate(Global_node)
    # acc = metrics_dict['test_acc']
    # f1 = metrics_dict['test_f1']
    # f_acc = metrics_dict['forget_acc']
    # r_error = metrics_dict['remain_error']
    # total_loss_t = metrics_dict['test_loss']
    recorder.printer(Global_node, rounds)


total_spend_time=time.time()-startt
print("Training time = {}".format(spend_time))
print("Total time = {}".format(total_spend_time))

        
if args.save_model:
    os.makedirs('server_saved_models', exist_ok=True)
    torch.save(Global_node.model.state_dict(), f'server_saved_models/train_type_{args.train_type}_{args.dataset}_model_seed{args.seed}_lr_{args.lr}_E{args.E}_misclient_{args.misclient}_mislabel_{args.mislabel}_sampler{args.sampler}_Dir{args.dir}_{args.local_model}_backdoor_{args.backdoor_attack}.pth')
