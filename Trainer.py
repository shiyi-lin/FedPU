import torch
import torch.nn as nn
from losses import decoupled_kd_loss, DistillKL
from torch.autograd import Variable
from tqdm import tqdm

KL_Loss = nn.KLDivLoss(reduction='batchmean')
Softmax = nn.Softmax(dim=1)
LogSoftmax = nn.LogSoftmax(dim=1)
CE_Loss = nn.CrossEntropyLoss()
KD_Loss = DistillKL(4)

def train_normal(node, round):
    node.model.to(node.device).train()
    train_loader = node.train_data
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.optimizer.zero_grad()
            epochs.set_description(description.format(node.num, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            output = node.model(data)
            target = target.view(-1).long()

            loss = CE_Loss(output, target)
            loss.backward()

            node.optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100


def train_grad(node, round):
    node.model.to(node.device).train()
    train_loader = node.train_data
    # Record initial weights
    w_old = {name: param.clone().detach() for name, param in node.model.named_parameters()}
    node.optimizer = torch.optim.SGD(node.model.parameters(), lr=node.args.lr, momentum=node.args.momentum, weight_decay=5e-4)

    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0

    description = "Node{:d}: loss={:.4f} acc={:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.optimizer.zero_grad()
            epochs.set_description(description.format(node.num, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            output = node.model(data)
            target = target.view(-1).long()

            loss = CE_Loss(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(node.model.parameters(), 5)

            node.optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100

    w_new = {name: param.clone().detach() for name, param in node.model.named_parameters()}
    delta_w = {name: w_new[name] - w_old[name] for name in w_new}
    node.grad = torch.cat([delta_w[name].flatten() for name in delta_w])


def train_unlearning(node, round):
    node.model.to(node.device).train()
    unlearning_loader = node.unlearning_data
    retraining_loader = node.retraining_data
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"
    
    w_old = {name: param.clone().detach() for name, param in node.model.named_parameters()}

    if round < node.args.max_unlearning:
        print(f"Unlearning phase with ul_lr: {node.args.ul_lr}")
        for param_group in node.optimizer.param_groups:
            param_group['lr'] = node.args.ul_lr
        with tqdm(unlearning_loader) as epochs:

            for idx, (data, target) in enumerate(epochs):
                node.optimizer.zero_grad()
                epochs.set_description(description.format(node.num, avg_loss, acc))
                data, target = data.to(node.device), target.to(node.device)
                target = target.view(-1).long()

                # target = target.squeeze()
                output = node.model(data)
                with torch.no_grad(): 
                
                    teacher_output = node.teacher_model(data)
                tckd, nckd = decoupled_kd_loss(
                    output, teacher_output, target,
                    node.args.tmp
                )
                loss = -node.args.kd_w * tckd.mean() + node.args.gamma * nckd.mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(node.model.parameters(), 5)
                node.optimizer.step()
                total_loss += loss.item()
                avg_loss = total_loss / (idx + 1)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum()
                acc = correct / len(unlearning_loader.dataset) * 100

    print(f"Retraining phase with rl_lr: {node.args.rl_lr}")
    for param_group in node.optimizer.param_groups:
        param_group['lr'] = node.args.rl_lr
  
    with tqdm(retraining_loader) as epochs:
        correct = 0.0
        total_loss = 0.0
        avg_loss = 0.0
        for idx, (data, target) in enumerate(epochs):
            node.optimizer.zero_grad()
            epochs.set_description(description.format(node.num, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            target = target.view(-1).long()

            # target = target.squeeze()

            output = node.model(data)
            with torch.no_grad(): 

                teacher_output = node.teacher_model(data)
            _, nckd = decoupled_kd_loss(
                output, teacher_output, target,
                node.args.tmp
            )

            loss_cls = CE_Loss(output, target)
            loss_div_decoupled = nckd.mean()
            loss = node.args.rce_w * loss_cls + node.args.div_w * loss_div_decoupled
           
            loss.backward()

            node.optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(retraining_loader.dataset) * 100


    
    w_new = {name: param.clone().detach() for name, param in node.model.named_parameters()}
    delta_w = {name: w_new[name] - w_old[name] for name in w_new}
    node.grad = torch.cat([delta_w[name].flatten() for name in delta_w])





class Trainer(object):

    def __init__(self, args):
        if args.algorithm == 'fedavg':
            self.train = train_normal
        elif args.algorithm == 'fedavggrad':
            self.train = train_grad
        elif args.algorithm == 'unlearning':
            self.train = train_unlearning   
        else:
            self.train = train_normal

    def __call__(self, node, round):
        self.train(node, round)






