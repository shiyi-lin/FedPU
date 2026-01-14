# FedPU

This is the official implementation of the paper:  
**"One-Shot Pareto-Optimal Federated Unlearning for Regulation-Aligned Medical A"**  

## 🚀 Quick Start
###  Install dependencies

'''bash
pip install -r requirements.txt
'''

### Download the dataset 
The dataset will be automatically downloaded to the `~/datasets` directory when you run the training script for the first time. If you prefer to store the dataset elsewhere, please manually set the path.

### 🔧 FL Training
Before performing unlearning, you need to train the base models. We provide two training modes to establish different experimental baselines:

#### 1. Normal FL Training (With Poisoned Forgetting Data)
This command trains a standard Federated Learning model where the target client's data contains backdoor triggers. This model serves as the **initial state** for the unlearning process.

```bash
python main.py --seed 42 --E 1 --dataset pathmnist --device cuda:1 --lr 0.01 --mislabel 0.2 --algorithm fedavg --local_model ResNet18 --backdoor_attack --train_type train  --lr_sch --save_model
```

#### 2. Retraining (Without Forgetting Data)
This command trains a model from scratch, completely excluding the data targeted for forgetting. 
```bash
python main.py --seed 2025 --E 1 --dataset pathmnist --device cuda:1 --lr 0.01 --mislabel 0.2 --algorithm fedavg --local_model ResNet18 --backdoor_attack --train_type unlearning --lr_sch --save_model
```

### 🔄 Unlearning Process
You can run the unlearning process with the following command:

```bash
python unlearning.py --seed 42 --E 1 --dataset pathmnist --device cuda:1 --rl_lr 0.0001 --ul_lr 0.00001 --mislabel 0.2 --algorithm unlearning --max_unlearning 10 --unlearning_round 10 --kd_w 0.2 --rce_w 1.0 --div_w 0.1 --du_w 0.8 --gamma 0.2 --local_model ResNet18 --backdoor_attack
```