import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Total
    parser.add_argument('--algorithm', type=str, default='fedavg',
                        help='Type of algorithms:{fedavg, fedavggrad, unlearning}')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--input_planes', type=int, default=3, help='input channels')

    parser.add_argument('--node_num', type=int, default=5,
                        help='Number of nodes')
    parser.add_argument('--R', type=int, default=100,
                        help='Number of communication rounds')
    parser.add_argument('--E', type=int, default=1,
                        help='Number of local epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed')
    

    parser.add_argument('--train_type', type=str, default='unlearning', choices=['train', 'unlearning'],
                        help='"train" for full dataset, "unlearning" to exclude data targeted for forgetting')
    parser.add_argument('--max_unlearning', type=int, default=5,
                        help='Rounds for the forgetting process')
    parser.add_argument('--unlearning_round', type=int, default=10,
                        help='Total unlearning_round')

    parser.add_argument('--tmp', type=int, default=4,
                        help='Temperature for kd')
    


    parser.add_argument('--kd_w', type=float, default=0.5,
                        help='Forget loss weight')
    parser.add_argument('--rce_w', type=float, default=1.0,
                        help='CE loss weight')
    parser.add_argument('--div_w', type=float, default=0.1,
                        help='Distillation loss weight')
    parser.add_argument('--du_w', type=float, default=0.8,
                        help='alpha for balancing')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='non target term weight')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='learning reate')
    # Model
    parser.add_argument('--global_model', type=str, default='ResNet18',
                        help='Type of global model: {LeNet5, ResNet18, mobilenetL, VitS}')
    parser.add_argument('--local_model', type=str, default='ResNet18',
                        help='Type of local model: {LeNet5, ResNet18, mobilenetL, VitS}')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='')
    
    # Data
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Name of the dataset')
    parser.add_argument('--batchsize', type=int, default=128,
                        help='batchsize')
    parser.add_argument('--split', type=int, default=5,
                        help='Data split')
    parser.add_argument('--dir', type=float, default=0.5,
                        help='Dirichlet distribution concentration parameter')

    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes in the dataset')
    parser.add_argument('--sampler', type=str, default='iid', choices=['iid', 'non-iid', 'dir'],
                        help='Data sampling strategy')
    parser.add_argument('--backdoor_attack', action='store_true', default=False,
                        help='Enable backdoor attack')

    parser.add_argument('--mislabel', type=float, default=0.5,
                    help='Ratio of mislabeled dat')

    parser.add_argument('--misclient', type=int, default=0,
                    help='The index of the client to be unlearned')
    parser.add_argument('--delete', action='store_true', default=False, 
                        help='delete mislabel data')

    # Optimizer Settings
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='Type of optimizer')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for normal training')
    parser.add_argument('--ul_lr', type=float, default=0.01,
                        help='Learning rate for the unlearning phase')
    parser.add_argument('--rl_lr', type=float, default=0.01,
                        help='Learning rate for the retain set training')
    parser.add_argument('--stop_decay', type=int, default=50,
                        help='')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--lr_sch', action='store_true', default=False,
                        help='Enable learning rate scheduler')
    args = parser.parse_args()
    return args
