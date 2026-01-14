import copy

import cvxopt
import Model
import numpy as np
import torch
from cvxopt import matrix
from vit_pytorch import SimpleViT


def init_model(model_type, num_classes, input_planes):
    if model_type == 'LeNet5':
        model = Model.LeNet5()
    elif model_type == 'LeNet':
        model = Model.LeNet(num_classes, input_planes)
    elif model_type == 'MLP':
        model = Model.MLP()
    elif model_type == 'ResNet18':
        model = Model.ResNet18(num_classes, input_planes)
    elif model_type == 'ResNetL18':
        model = Model.ResNetL18(num_classes, input_planes)
    elif model_type == 'CNN':
        model = Model.CNN()
    elif model_type == 'FLNet':
        model = Model.FLNet()
    elif model_type == 'VitS':
        model = SimpleViT(
            image_size = 32,
            patch_size = 4,
            num_classes = num_classes,
            dim = 512,
            depth = 6,
            heads = 16,
            mlp_dim = 128
        )
    elif model_type == "mobilenetL":
        from mobilenetv3 import MobileNetV3_Large
        model = MobileNetV3_Large(in_channels=input_planes, num_classes=num_classes, featdim=512)

    return model


def init_optimizer(model, args):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    return optimizer

def weights_zero(model):
    for p in model.parameters():
        if p.data is not None:
            p.data.detach_()
            p.data.zero_()


class Node(object):
    def __init__(self, num, train_data, test_data, args):
        self.args = args
        self.num = num + 1
        self.device = self.args.device
        self.train_data = train_data
        self.test_data = test_data
        self.unlearning_data = None
        self.retraining_data = None


        self.model = init_model(self.args.local_model, self.args.num_classes, self.args.input_planes).to(self.device)
        self.teacher_model = init_model(self.args.local_model, self.args.num_classes, self.args.input_planes).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)
        self.loss = []
        self.grad = []
        # self.update = [] 


    def fork(self, global_node):
        # fork new global_model
        self.model = copy.deepcopy(global_node.model).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)


    
class Global_Node(object):
    def __init__(self, test_data, args):
        self.num = 0
        self.args = args
        self.device = self.args.device
        self.model = init_model(self.args.global_model, self.args.num_classes, self.args.input_planes).to(self.device)

        self.test_data = test_data
        self.Dict = self.model.state_dict()
        self.unlearning_data = None
        self.retraining_data = None
        self.init = False

    def merge(self, Edge_nodes):

        Node_State_List = [copy.deepcopy(Edge_nodes[i].model.state_dict()) for i in range(len(Edge_nodes))]
        self.Dict = Node_State_List[0]

        for key in self.Dict.keys():
            for i in range(1, len(Node_State_List)):
                self.Dict[key] += Node_State_List[i][key]
                
            self.Dict[key] = self.Dict[key].float()  
            self.Dict[key] /= len(Node_State_List)
        self.model.load_state_dict(self.Dict)
   

    def get_nearest_oth_d(self, gr_locals, gu):
        A = gr_locals
        
        A_T = A.T
        c = gu
        
        AAT_1 = self.cal_psedoinverse(A @ A_T)  
        
        Ac= A @ c.reshape(-1, 1)
        
        AAT_1_Ac = AAT_1 @ Ac
        
        d = c - (A_T @ AAT_1_Ac).reshape(-1)

        return d

    def cal_psedoinverse(self, matrix):
        U, s, V = torch.svd(matrix)  
        primary_sigma_indices = torch.where(s >= 1e-6)[0]
        s[primary_sigma_indices] = 1 / s[primary_sigma_indices]
        S = torch.diag(s)
        psedoinverse = V @ S @ U.T
        return psedoinverse

    def merge_pareto(self, Edge_nodes, rounds):

        g_forget = Edge_nodes[self.args.misclient].grad
        grad_norm = torch.norm(g_forget) + 1e-12
        g_locals = torch.stack([Edge_nodes[i].grad for i in range(len(Edge_nodes))])
        # cal pareto gradient with all local gradients
        d_pareto, _ = get_pareto_gradient(
            g_locals, self.device
        )

        d_pareto = d_pareto*grad_norm
        d_combined = self.args.du_w * g_forget + (1 - self.args.du_w)*d_pareto
        
        server_initial_state_dict = self.model.state_dict()

        client_final_state_dict = Edge_nodes[self.args.misclient].model.state_dict()

        target_state_dict = {}
        param_idx = 0
        for key, param in server_initial_state_dict.items():
            if key in dict(self.model.named_parameters()):
                num_param = param.numel()
                update_slice = d_combined[param_idx : param_idx + num_param].view_as(param)
                target_state_dict[key] = server_initial_state_dict[key] + self.args.eta*update_slice
                param_idx += num_param
            else:

                target_state_dict[key] = client_final_state_dict[key]

        self.model.load_state_dict(target_state_dict) 


def get_pareto_gradient(grads, device):
    """Normalizes gradients and solves the Quadratic Programming problem for MGDA."""
    
    norm_vec = torch.norm(grads, dim=1)
    grads = grads / (norm_vec.reshape(-1, 1) + 1e-6)
    sol, _ = setup_qp_and_solve(grads.cpu().detach().numpy())
    sol = torch.from_numpy(sol).float().to(device)
    d_pareto = sol @ grads

    return d_pareto, grads
     

def setup_qp_and_solve(vec):
  
    P = np.dot(vec, vec.T)
    n = P.shape[0]
    q = np.zeros(n)
    G = - np.eye(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    cvxopt.solvers.options['show_progress'] = False
    sol, optimal_flag = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol, optimal_flag

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = 0.5 * (P + P.T)  
    P = P.astype(np.double)
    q = q.astype(np.double)
    args = [matrix(P), matrix(q)]  
    if G is not None:
        args.extend([matrix(G), matrix(h)])  
        if A is not None:
            args.extend([matrix(A), matrix(b)])  
    sol = cvxopt.solvers.qp(*args)
    optimal_flag = 1
    if 'optimal' not in sol['status']:
        optimal_flag = 0
    return np.array(sol['x']).reshape((P.shape[1],)), optimal_flag

