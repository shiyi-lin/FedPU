import copy
import datetime
import time

import torch
from Args import args_parser
from Data import Data as Data_g
from Node import Global_Node, Node
from Trainer import Trainer
from utils import Recorder, Summary, fix_seed

# Initialize args
args = args_parser()
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
args.global_model = args.local_model
CONSTANT_ALGO = args.algorithm
Train_ul = Trainer(args)

args.algorithm = 'fedavggrad'
Train_normal = Trainer(args)
args.algorithm = CONSTANT_ALGO

recorder = Recorder(args)
Summary(args)
print('Running on', args.device)

# Fix Seed
fix_seed(args.seed)
Data = Data_g(args)

current_time = datetime.datetime.now()
# Initialize Nodes
Global_node = Global_Node(Data.test_loader, args)
Edge_nodes = [Node(k, Data.train_loader[k], Data.test_loader, args) for k in range(args.node_num)]


# Get unlearning and retraining samples
Edge_nodes[args.misclient].unlearning_data = Data.unlearning_loader
Edge_nodes[args.misclient].retraining_data = Data.retraining_loader
Global_node.unlearning_data_eval = Data.unlearning_loader_eval
Global_node.retraining_data_eval = Data.retraining_loader_eval
Global_node.unlearning_data = Data.unlearning_loader
Global_node.retraining_data = Data.retraining_loader

if args.sampler == 'iid':
    foldpth = '/data_a/lsy/FedAVG/server_saved_models_sup'
else:
    foldpth = '/data_a/lsy/FedAVG/server_saved_models_sup_dir_origin'
# Load original model
if args.dataset == 'pathmnist' and args.sampler == 'dir':
    modelpth = f'{foldpth}/{args.dataset}_model_seed{args.seed}_lr_0.01_E{args.E}_misclient_{args.misclient}_mislabel_{args.mislabel}_sampler{args.sampler}_Dir{args.dir}_{args.local_model}_nodenum{args.node_num}_backdoor9_{args.backdoor_attack}_1127N_0.pth'
else:
    modelpth = f'{foldpth}/{args.dataset}_model_seed{args.seed}_lr_0.01_E{args.E}_misclient_{args.misclient}_mislabel_{args.mislabel}_sampler{args.sampler}_Dir{args.dir}_{args.local_model}_nodenum{args.node_num}_backdoor9_{args.backdoor_attack}_1121N_0.pth'

print(modelpth)
Global_node.model.load_state_dict(torch.load(modelpth,  map_location=args.device))


startt = time.time()

metrics_dict = recorder.validate(Global_node)
recorder.printer(Global_node, 0)

# Get latest gradients from normal training
for k in range(len(Edge_nodes)):
    Edge_nodes[k].fork(Global_node)          
    for epoch in range(args.E):
        Train_normal(Edge_nodes[k], 0)


spend_time = 0.0


Edge_nodes[args.misclient].teacher_model = copy.deepcopy(Global_node.model)
Edge_nodes[args.misclient].fork(Global_node)   

# Unlearning Loop
for rounds in range(args.unlearning_round):
    print('===============Unlearning, the {:d}-th round==============='.format(rounds + 1))
    roundtime1 = time.time()
   
    for epoch in range(args.E):
        Train_ul(Edge_nodes[args.misclient], rounds)

    print('--------------------------------')

    Global_node.merge_pareto(Edge_nodes, rounds)   
        
    Edge_nodes[args.misclient].fork(Global_node)  

    
    roundtime = time.time() - roundtime1
    spend_time += roundtime 
    #Test
    metrics_dict = recorder.validate(Global_node)
    recorder.printer(Global_node, rounds)


total_spend_time=time.time()-startt
print("Unlearning time = {}".format(spend_time))
print("Total time = {}".format(total_spend_time))




Summary(args)
