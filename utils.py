import torch
import numpy as np
from torch import nn
import random
from sklearn.metrics import f1_score

class Recorder(object):
    """Tracks and logs performance metrics for each node and the global server."""
    def __init__(self, args):
        self.args = args
        self.counter = 0
  
        self.val_loss = {}
        self.val_acc = {}
        self.f1s = {}
        self.forget_acc = {}
        self.remain_acc = {}

        for i in range(self.args.node_num + 1):
            self.val_loss[str(i)] = []
            self.val_acc[str(i)] = []
            self.f1s[str(i)] = []
            self.forget_acc[str(i)] = []
            self.remain_acc[str(i)] = []

    def validate(self, node):
 
        self.counter += 1
        
        test_metrics = _evaluate_on_loader(node.model, node.test_data, node.device, self.args.num_classes)

        print(f"\n--- Validation Round {self.counter} for Node {node.num} ---")


        forget_metrics = _evaluate_on_loader(node.model, node.unlearning_data_eval, node.device)
        
        remain_metrics = _evaluate_on_loader(node.model, node.retraining_data_eval, node.device)

        self.forget_acc[str(node.num)].append(forget_metrics['acc'])
        self.remain_acc[str(node.num)].append(remain_metrics['acc'])

        self.val_acc[str(node.num)].append(test_metrics['acc'])
        self.val_loss[str(node.num)].append(test_metrics['loss'])
        self.prec = test_metrics['precision_per_class']
        return {
            'test_acc': test_metrics['acc'],
            'test_f1': test_metrics['f1'],
            'test_loss': test_metrics['loss'],
            'forget_acc': forget_metrics['acc'],
            'remain_error': round(100 - remain_metrics['acc'], 4)
        }

    def printer(self, node, rounds):

        if node.num == 0:
            print('Test Acc on server:',self.val_acc[str(node.num)])
            print('Forget Acc on server:',self.forget_acc[str(node.num)])
            print('Remain Acc on server:',self.remain_acc[str(node.num)])

            val_loss_list = [round(loss_value, 2) for loss_value in self.val_loss[str(node.num)]]
            print('loss:', val_loss_list)
            print('Acc on each classes:', self.prec)

     
        else:
            print('Node Test Acc:', self.val_acc[str(node.num)][rounds])
            if node.num==self.args.misclient+1:
                print('Node Forget Acc:',self.forget_acc[str(node.num)])
                print('Node Remain Acc:',self.remain_acc[str(node.num)])

            print('Acc on each classes:', [round(x*100, 2) for x in self.prec])


def LR_scheduler(rounds, Edge_nodes, args):

    for i in range(len(Edge_nodes)):
        Edge_nodes[i].args.lr = args.lr
        Edge_nodes[i].optimizer.param_groups[0]['lr'] = args.lr
    
    print('Learning rate={:.4f}'.format(args.lr))



def Summary(args):
    print("Summary:\n")
    print("algorithm:{}\n".format(args.algorithm))
    print("seed:{}\n".format(args.seed))
    print("dataset:{}\tbatchsize:{}\n".format(args.dataset, args.batchsize))
    print("node_num:{}\n".format(args.node_num))
    print("global epochs:{},\tlocal epochs:{},\n".format(args.R, args.E))
    print("global_model:{}, \tlocal model:{},\n".format(args.global_model, args.local_model))
    print("backdoor_attack:{},\tlr_sch:{},\n".format(args.backdoor_attack, args.lr_sch))
    print("train_type:{},\tsave model:{},\n".format(args.train_type, args.save_model))

def _evaluate_on_loader(model, loader, device, num_classes=None):
    model.to(device).eval()
    total_loss = 0.0
    correct = 0
    all_preds = []
    all_targets = []

    if len(loader.dataset) == 0:
        return {'loss': 0, 'acc': 0, 'f1': 0, 'precision_per_class': [0]*num_classes if num_classes else []}

    criterion = nn.CrossEntropyLoss(reduction='sum') # 使用 'sum' 方便最后除以总数
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            target = target.view(-1).long()
            
            output = model(data)
            
            total_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = (correct / len(loader.dataset)) * 100
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    
    prec_per_class = []
    if num_classes:
        for i in range(num_classes):
            mask = all_targets == i
            if mask.sum() > 0:
                class_correct = (all_preds[mask] == all_targets[mask]).sum().item()
                prec = (class_correct / mask.sum().item()) * 100
                prec_per_class.append(prec)
            else:
                prec_per_class.append(0)

    return {
        'loss': round(avg_loss, 4),
        'acc': round(accuracy, 4),
        'f1': round(f1, 4),
        'precision_per_class': [round(p, 4) for p in prec_per_class]
    }

class AverageMeter(object):
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
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
 
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
