import random
import medmnist
import numpy as np
import torch
from medmnist import INFO
from PIL import Image

from torch.utils.data import DataLoader, Dataset, Subset, random_split, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import torch.nn.functional as F
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd

medmnist_data = ["pathmnist", "chestmnist", "dermamnist", "octmnist", "pneumoniamnist", "retinamnist", "breastmnist",
        "bloodmnist", 'tissuemnist', "organamnist", "organcmnist", "organsmnist", "organmnist3d", "nodulemnist3d",
        "adrenalmnist3d", "fracturemnist3d", "vesselmnist3d", "synapsemnist3d"]

def print_client_data_distribution(splited_trainset, num_classes):
    """Logs the number of samples per class for each client."""
    print("=" * 50)
    print(" " * 15 + "Client Data Distribution")
    print("=" * 50)
    
    header = f"| {'Client ID':<10} | {'Total':<8} |"
    for i in range(num_classes):
        header += f" C{i:<3} |"
    print(header)
    print("-" * len(header))

    overall_distribution = np.zeros(num_classes, dtype=int)

    for client_id, client_dataset in enumerate(splited_trainset):
        
        if isinstance(client_dataset, Subset):
 
            targets = np.array(client_dataset.dataset.targets)
            client_targets = targets[client_dataset.indices]

        elif hasattr(client_dataset, 'targets'):
            client_targets = np.array(client_dataset.targets)
        else:
            print(f"Warning: Client {client_id} has an unsupported dataset type: {type(client_dataset)}")
            continue

        total_samples = len(client_targets)
        class_counts = np.bincount(client_targets, minlength=num_classes).astype(int)
        overall_distribution += class_counts
        
        row_str = f"| {client_id:<10} | {total_samples:<8} |"
        for count in class_counts:
            row_str += f" {count:<4} |"
        print(row_str)

    print("-" * len(header))
    
    total_str = f"| {'Total':<10} | {overall_distribution.sum():<8} |"
    for count in overall_distribution:
        total_str += f" {count:<4} |"
    print(total_str)
    print("=" * len(header)) 

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):

        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]
        target = self.targets[idx]

        if isinstance(sample, np.ndarray):
            sample = Image.fromarray(sample)
        else:
            sample = Image.fromarray(sample.numpy())

        if self.transform:
            sample = self.transform(sample)

        return sample, target

def Datasets(args):
    trainset, testset = None, None
    print(args.dataset)
    if args.dataset == 'cifar10':
        args.num_classes = 10
        tra_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = CIFAR10(root="~/datasets", train=True, download=True, transform=tra_trans)
        testset = CIFAR10(root="~/datasets", train=False, download=True, transform=val_trans)

    elif args.dataset == 'mnist':
        args.num_classes = 10
        args.input_planes = 1
        tra_trans = transforms.Compose([
            transforms.Pad(2, padding_mode='edge'),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        val_trans = transforms.Compose([
            transforms.Pad(2, padding_mode='edge'),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = MNIST(root='~/datasets', train=True, transform=tra_trans, download= True)
        testset = MNIST(root='~/datasets', train=False, transform=val_trans, download= True)
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        tra_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
        ])
        val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
        ])
        trainset = CIFAR100(root="~/datasets", train=True, download=True, transform=tra_trans)
        testset = CIFAR100(root="~/datasets", train=False, download=True, transform=val_trans)
    elif args.dataset in medmnist_data:
        info = INFO[args.dataset]
        DataClass = getattr(medmnist, info['python_class'])

        args.num_classes = len(info['label'])
        if args.dataset in ["tissuemnist", "chestmnist", "octmnist", "breastmnist", "pneumoniamnist", "organamnist"]:
            args.input_planes = 1
        print('input_plane:', args.input_planes)

        tra_trans = transforms.Compose([
            transforms.Pad(2, padding_mode='edge'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
        ])
        val_trans = transforms.Compose([
            transforms.Pad(2, padding_mode='edge'),

            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
        ])
        trainset = DataClass(split='train', transform=tra_trans, download=True)
        testset = DataClass(split='test', transform=val_trans, download=True)
        print("size:", trainset.labels.shape)
        trainset.targets = trainset.labels.squeeze(1)
        testset.targets = testset.labels.squeeze(1)
    
    return trainset, testset



class Data(object):

    def __init__(self, args):
        self.args = args
        self.trainset, self.testset = None, None
        self.unlearning_indices = []
        self.retrain_indices = []
        set_seeds(args.seed)

        # Load and Partition Data
        trainset, testset = Datasets(args)
        splited_trainset = self.partition_data(trainset)
        print_client_data_distribution(splited_trainset, self.args.num_classes)

        
        client1_dataidx = splited_trainset[self.args.misclient].indices           #
        k = args.mislabel                                           
        print('target client before:',len(client1_dataidx))

        if self.args.backdoor_attack:
            backdoor = PoisoningAttackBackdoor(add_pattern_bd)
                
            poisoned_client_dataset, self.unlearning_set, unlearning_dataset_eval, self.retrain_indices = create_backdoor_sets(
                trainset,
                client1_dataidx,
                k,
                self.args.num_classes,
                backdoor,
                trainset.transform,
                testset.transform,
                args
            )
            print('Poisoned samples num:', len(self.unlearning_set.targets))

            # Phase Control: If 'train', target client uses poisoned data. If 'unlearning', target client uses Dr.
            if args.train_type == 'train':
                splited_trainset[self.args.misclient] = poisoned_client_dataset
                print('Client dataset size after poisoning:', len(splited_trainset[self.args.misclient].targets))
            else: 
                splited_trainset[self.args.misclient] = Subset(trainset, self.retrain_indices)
                print('Client dataset size for retraining:', len(splited_trainset[self.args.misclient].indices))
                
            self.retraining_set = Subset(trainset, self.retrain_indices)
            
            if args.dataset in medmnist_data:
                original_data = trainset.imgs
            else:
                original_data = trainset.data
            original_targets = np.array(trainset.targets)
            retraining_dataset_eval = CustomDataset(
                data=original_data[self.retrain_indices],
                targets=original_targets[self.retrain_indices],
                transform=testset.transform
            )
        
                
        else:
            num_to_change = int(len(client1_dataidx) * k)
            unlearning_indices = np.random.choice(client1_dataidx, num_to_change, replace=False).tolist()
            self.retrain_indices = [idx for idx in client1_dataidx if idx not in unlearning_indices]
            if args.dataset in medmnist_data:
                original_data = trainset.imgs
            else:
                original_data = trainset.data
            original_targets = np.array(trainset.targets)

            unlearning_data = original_data[unlearning_indices]
            unlearning_targets = np.array(original_targets)[unlearning_indices].astype(np.int64)
            self.unlearning_set = CustomDataset(
                data=unlearning_data,
                targets=unlearning_targets,
                transform=trainset.transform
            )
            unlearning_dataset_eval = CustomDataset(
                data=unlearning_data,
                targets=unlearning_targets,
                transform=testset.transform
            )

            self.retraining_set = Subset(trainset, self.retrain_indices)

            if args.train_type == 'train':
                client_train_data = original_data[client1_dataidx]
                client_train_targets = np.array(original_targets)[client1_dataidx].astype(np.int64)
                splited_trainset[self.args.misclient] = CustomDataset(
                    data=client_train_data,
                    targets=client_train_targets,
                    transform=trainset.transform
                )
            
            else: 

                splited_trainset[self.args.misclient] = self.retraining_set
            original_targets = np.array(trainset.targets)

            retraining_dataset_eval = CustomDataset(
                data=original_data[self.retrain_indices],
                targets=original_targets[self.retrain_indices],
                transform=testset.transform
            )
            print('Unlearning set size:', len(self.unlearning_set.targets))
            print('Client remaining set size:', len(self.retraining_set))
            print('The sample indices deleted are:', unlearning_indices[:10])
            print('Training set size:', len(splited_trainset[self.args.misclient]))

            

        self.unlearning_loader = DataLoader(self.unlearning_set, batch_size=args.batchsize, shuffle=True, num_workers=4)
        self.retraining_loader = DataLoader(self.retraining_set, batch_size=args.batchsize, shuffle=True, num_workers=4)
        self.unlearning_loader_eval = DataLoader(unlearning_dataset_eval, batch_size=args.batchsize, shuffle=False, num_workers=4)
        self.retraining_loader_eval = DataLoader(retraining_dataset_eval, batch_size=args.batchsize, shuffle=False, num_workers=4)
        
        self.train_loader = [DataLoader(splited_trainset[i], batch_size=args.batchsize, shuffle=True, num_workers=4)
                             for i in range(args.node_num)]
        self.test_loader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=4)

    def partition_data(self, dataset: Dataset):
        if self.args.sampler == 'iid':
            return self._iid_partition(dataset)
        elif self.args.sampler == 'dir':
            return self._dirichlet_partition(dataset)
    def _iid_partition(self, dataset: Dataset) -> list:
        n_items = [len(dataset) // self.args.node_num] * self.args.node_num
        n_items[-1] += len(dataset) % self.args.node_num
        return random_split(dataset, n_items, generator=torch.Generator().manual_seed(self.args.seed))

    def _dirichlet_partition(self, dataset: Dataset) -> list[Subset]:
        targets = np.array(dataset.targets)
        num_classes = len(np.unique(targets))
        num_clients = self.args.node_num
        alpha = self.args.dir  

        total_data = len(dataset)
        base_size = total_data // num_clients
        client_sizes = np.full(num_clients, base_size)
        client_sizes[-1] += total_data % num_clients

        label_proportions = np.random.dirichlet(
            np.full(num_classes, alpha), 
            size=num_clients
        )

        label_indices = {}
        for label in range(num_classes):
            label_indices[label] = np.where(targets == label)[0].tolist()
            np.random.shuffle(label_indices[label])

        client_indices = [[] for _ in range(num_clients)]
        
        for client_id in range(num_clients):
            proportions = label_proportions[client_id]
            counts = np.floor(proportions * client_sizes[client_id]).astype(int)
            
            for label in range(num_classes):
                if counts[label] > 0 and len(label_indices[label]) >= counts[label]:
                    selected = label_indices[label][:counts[label]]
                    label_indices[label] = label_indices[label][counts[label]:]
                    client_indices[client_id].extend(selected)
        
        remaining_samples = []
        for label in range(num_classes):
            remaining_samples.extend(label_indices[label])
        
        np.random.shuffle(remaining_samples)
        
        for client_id in range(num_clients):
            samples_needed = client_sizes[client_id] - len(client_indices[client_id])
            if samples_needed > 0 and samples_needed <= len(remaining_samples):
                client_indices[client_id].extend(remaining_samples[:samples_needed])
                remaining_samples = remaining_samples[samples_needed:]
        
        for client_id in range(num_clients):
            while len(client_indices[client_id]) < client_sizes[client_id] and remaining_samples:
                client_indices[client_id].append(remaining_samples.pop(0))
            
            if len(client_indices[client_id]) > client_sizes[client_id]:
                excess = len(client_indices[client_id]) - client_sizes[client_id]
                removed = client_indices[client_id][-excess:]
                client_indices[client_id] = client_indices[client_id][:-excess]
                remaining_samples.extend(removed)
            
            np.random.shuffle(client_indices[client_id])
            
            if len(client_indices[client_id]) != client_sizes[client_id]:
                print(f"Warning: Client {client_id}'s sample number is {len(client_indices[client_id])}, expecting {client_sizes[client_id]}")
        
        return [Subset(dataset, indices) for indices in client_indices]
   
def create_backdoor_sets(base_trainset, client_indices, k, num_classes, backdoor_attack_fn, transform, eval_transform, args):
    """Injects triggers into a subset of client data and returns separated Df and Dr."""
    if args.dataset in medmnist_data:
        raw_data = base_trainset.imgs[client_indices]
    else:
        raw_data = base_trainset.data[client_indices]
    if isinstance(raw_data, torch.Tensor):
        client_data = raw_data.cpu().numpy()
    else:
        client_data = np.array(raw_data)

    client_labels = np.array(base_trainset.targets)[client_indices]

    # Normalize data to [0, 1]
    if client_data.max() > 1.0:
        client_data_float = client_data.astype(np.float32) / 255.0
    else:
        client_data_float = client_data.astype(np.float32)

    target_label = num_classes - 1
    poison_candidate_local_indices = np.where(client_labels != target_label)[0]
    
    num_to_poison = int(len(poison_candidate_local_indices) * k)
    if num_to_poison == 0:
        print("Warning: No samples to poison for the target client.")
        client_dataset = Subset(base_trainset, client_indices)
        empty_unlearning_set = CustomDataset(np.array([]), np.array([]), transform)
        return client_dataset, empty_unlearning_set, client_indices

    poison_selected_local_indices = np.random.choice(poison_candidate_local_indices, num_to_poison, replace=False)
    
    # Execute Poisoning via ART
    data_to_poison = client_data_float[poison_selected_local_indices]
    
    one_hot_target = F.one_hot(torch.tensor(target_label), num_classes=num_classes).numpy()
    poisoned_data_float, _ = backdoor_attack_fn.poison(data_to_poison, y=one_hot_target, broadcast=True)
    poisoned_labels = np.full(num_to_poison, target_label)

    final_client_data_float = np.copy(client_data_float)
    final_client_labels = np.copy(client_labels)
    final_client_data_float[poison_selected_local_indices] = poisoned_data_float
    final_client_labels[poison_selected_local_indices] = poisoned_labels

    final_client_data_uint8 = (final_client_data_float * 255).clip(0, 255).astype(np.uint8)
    poisoned_client_dataset = CustomDataset(
        data=final_client_data_uint8,
        targets=final_client_labels,
        transform=transform
    )
    # Reconstruct client dataset (Dr + Df_poisoned)
    poisoned_data_uint8 = (poisoned_data_float * 255).clip(0, 255).astype(np.uint8)
    unlearning_set = CustomDataset(
        data=poisoned_data_uint8,
        targets=poisoned_labels,
        transform=transform
    )
    unlearning_set_eval = CustomDataset(
        data=poisoned_data_uint8,
        targets=poisoned_labels,
        transform=eval_transform
    )

    client_indices_array = np.array(client_indices)
    poison_selected_global_indices = client_indices_array[poison_selected_local_indices]
    retrain_indices = list(set(client_indices) - set(poison_selected_global_indices))
    print('The sample indices poisoned are:', poison_selected_local_indices[:10])

    return poisoned_client_dataset, unlearning_set, unlearning_set_eval, retrain_indices

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
