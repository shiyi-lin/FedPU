import torch.nn as nn
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(
                inchannel,
                outchannel,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    inchannel, outchannel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, residual_block, num_classes=10, input_planes=3):
        super(ResNet, self).__init__()
        self.Loc_reshape_list = None
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_planes, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(residual_block, 64, 2, stride=1)
        self.layer2 = self.make_layer(residual_block, 128, 2, stride=2)
        self.layer3 = self.make_layer(residual_block, 256, 2, stride=2)
        self.layer4 = self.make_layer(residual_block, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
        # self.create_Loc_reshape_list()

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def create_Loc_reshape_list(self):
        currentIdx = 0
        self.Loc_reshape_list = []
        for i, p in enumerate(self.parameters()):
            flat = p.data.clone().flatten()
            self.Loc_reshape_list.append(torch.arange(currentIdx, currentIdx + len(flat), 1).reshape(p.data.shape))  
            currentIdx += len(flat)
        


    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

class ResNetL(nn.Module):
    def __init__(self, residual_block, num_classes=10, input_planes=3):
        super(ResNetL, self).__init__()
        self.Loc_reshape_list = None
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_planes, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer1 = self.make_layer(residual_block, 64, 2, stride=1)
        self.layer2 = self.make_layer(residual_block, 128, 2, stride=2)
        self.layer3 = self.make_layer(residual_block, 256, 2, stride=2)
        self.layer4 = self.make_layer(residual_block, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
        self.create_Loc_reshape_list()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def create_Loc_reshape_list(self):
        currentIdx = 0
        self.Loc_reshape_list = []
        for i, p in enumerate(self.parameters()):
            flat = p.data.clone().flatten()
            self.Loc_reshape_list.append(torch.arange(currentIdx, currentIdx + len(flat), 1).reshape(p.data.shape))  
            currentIdx += len(flat)
        


    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18(num_classes, input_planes):
    return ResNet(ResidualBlock, num_classes, input_planes)
def ResNetL18(num_classes, input_planes):
    return ResNetL(ResidualBlock, num_classes, input_planes)
class ResNetUL(nn.Module):
    def __init__(self, residual_block, num_classes=10, input_planes=3):
        super(ResNetUL, self).__init__()
        self.Loc_reshape_list = None
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_planes, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(residual_block, 64, 2, stride=1)
        self.layer2 = self.make_layer(residual_block, 128, 2, stride=2)
        self.layer3 = self.make_layer(residual_block, 256, 2, stride=2)
        self.layer4 = self.make_layer(residual_block, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
        self.fc_ul = nn.Linear(512, num_classes)

        self.create_Loc_reshape_list()

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def create_Loc_reshape_list(self):
        currentIdx = 0
        self.Loc_reshape_list = []
        for i, p in enumerate(self.parameters()):
            flat = p.data.clone().flatten()
            self.Loc_reshape_list.append(torch.arange(currentIdx, currentIdx + len(flat), 1).reshape(p.data.shape))  
            currentIdx += len(flat)
        


    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)


        # out = self.fc(out)
        out_ul = self.fc_ul(out)
        # z = torch.cat((out,out_ul),dim=1)

        return out_ul
    
def ResNet18_ul(num_classes, input_planes):
    return ResNetUL(ResidualBlock, num_classes, input_planes)

class LeNet5(nn.Module):
    def __init__(self, num_classes=10, input_planes=3):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet(nn.Module):
    def __init__(self, num_classes=10, input_planes=3):
        super(LeNet, self).__init__()
        self.Loc_reshape_list = None
        self.conv1 = nn.Conv2d(input_planes, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.create_Loc_reshape_list()

    def create_Loc_reshape_list(self):
        currentIdx = 0
        self.Loc_reshape_list = []
        for i, p in enumerate(self.parameters()):
            flat = p.data.clone().flatten()
            self.Loc_reshape_list.append(torch.arange(currentIdx, currentIdx + len(flat), 1).reshape(p.data.shape))
            currentIdx += len(flat)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2) 
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
   
class LeNetUL(nn.Module):
    def __init__(self, num_classes=10, input_planes=3):
        super(LeNetUL, self).__init__()
        self.Loc_reshape_list = None
        self.conv1 = nn.Conv2d(input_planes, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.fc3_ul = nn.Linear(84, num_classes)

        self.create_Loc_reshape_list()

    def create_Loc_reshape_list(self):
        currentIdx = 0
        self.Loc_reshape_list = []
        for i, p in enumerate(self.parameters()):
            flat = p.data.clone().flatten()
            self.Loc_reshape_list.append(torch.arange(currentIdx, currentIdx + len(flat), 1).reshape(p.data.shape))
            currentIdx += len(flat)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3_ul(x)
        return x
class MLP(nn.Module):
    def __init__(self, num_classes=10, input_planes=3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_planes * 32 * 32, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)
        self.input_planes = input_planes

    def forward(self, x):
        x = x.view(-1, self.input_planes * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes=10, input_planes=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class FLNet(nn.Module):
    def __init__(self):
        super(FLNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x