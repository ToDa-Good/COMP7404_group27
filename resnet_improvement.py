import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

#original resnet with 2 conv layerss
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_channel != self.expansion*out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, self.expansion*out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channel)
            )

    def forward(self, x):
        out =torch.relu(self.bn1(self.conv1(x)))
        out =self.bn2(self.conv2(out))
        identity=self.shortcut(x)
        out += identity
        return torch.relu(out)
    
class ResNeXtBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, cardinality=32):
        super().__init__()

        group_width = out_channel//cardinality #calcuate conv channel num of each part
        #group_width = out_channel/cardinality(such as out_channel=64, cardinality=32 which means we have 2 conv channel)

        #then use grouped conv in each layer
        #[Conv1x1]->[BN]->ReLU->[GroupedConv3x3]->[BN]->ReLU->[Conv1x1]->[BN]
        self.conv1 = nn.Conv2d(in_channel, group_width * cardinality, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width*cardinality)
        
        self.conv2 = nn.Conv2d(group_width *cardinality, group_width *cardinality, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width * cardinality)
        
        self.conv3 = nn.Conv2d(group_width*cardinality, out_channel*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel*self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel*self.expansion)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        identity=self.shortcut(x)
        out += identity
        return torch.relu(out)
    
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        #reduction define how much to reduce the fully connet layers
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), #reduction
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),  #recover
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ =x.size() #[batch, channels, height, width]

        y=self.avg_pool(x).view(b, c) #Squeeze
        y=self.fc(y).view(b, c, 1, 1) #Excitation
        
        output=x*y.expand_as(x) #make the connection between weight and original data
        return output 

class SEResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        #Squeeze-and-Excitation
        self.se=SEBlock(out_channel)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != self.expansion*out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, self.expansion*out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channel)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return torch.relu(out)
    
class StochasticDepth(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p #survival probability, which means each have only 50% to be survive
        
    def forward(self, x):
        if not self.training:
            return x
        
        #generate true or false randomly(each have 50% possibility)    
        binary_tensor = torch.rand(x.size(0), 1, 1, 1, device=x.device)<self.p
        output=x*binary_tensor/self.p #keep E (should /p)
        return output

class SDResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, p=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        #give the random reduce(p is the possiblity)
        self.sd=StochasticDepth(p)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != self.expansion*out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, self.expansion*out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channel)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #randomly reduce the way
        out = self.sd(out)
        identity=self.shortcut(x)
        out += identity
        return torch.relu(out)
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, **kwargs):
        #we use block to define the model(which resnet) we choose
        #we use kwargs to give different variables in different resnet(such as cardinality and reduction)
        #for num_blocks we use [2,2,2,2] to use resnet18
        super().__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, **kwargs)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, out_channel, num_blocks, stride, **kwargs):
        #for different resnet, generate different layer
        strides = [stride] + [1]*(num_blocks-1) #generate a list of strides
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride, **kwargs))
            self.in_channel = out_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss/len(loader), 100.*correct/total

def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss/len(loader), 100.*correct/total

#[2,2,2,2] means resnet18
models = {
    "Original ResNet": ResNet(BasicBlock, [2, 2, 2, 2]),
    "ResNeXt": ResNet(ResNeXtBlock, [2, 2, 2, 2], cardinality=32),
    "SE-ResNet": ResNet(SEResNetBlock, [2, 2, 2, 2]),
    "Stochastic Depth ResNet": ResNet(SDResNetBlock, [2, 2, 2, 2], p=0.5)
}

results = {name: {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []} 
           for name in models.keys()}

epochs = 50
criterion = nn.CrossEntropyLoss()

for name, model in models.items():
    print(f"\nTraining {name}...")
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    
    for epoch in range(epochs):
        train_loss,train_acc=train(model,train_loader,optimizer,criterion)
        test_loss, test_acc= test(model,test_loader, criterion)
        
        scheduler.step()
        
        results[name]["train_loss"].append(train_loss)
        results[name]["train_acc"].append(train_acc)
        results[name]["test_loss"].append(test_loss)
        results[name]["test_acc"].append(test_acc)
        
        #give the result
        print(f"Epoch {epoch+1}/{epochs} | Test Acc: {test_acc:.2f}%")

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
for name in models.keys():
    plt.plot(results[name]["test_acc"], label=name)
plt.title("Test Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.subplot(1, 2, 2)
for name in models.keys():
    plt.plot(results[name]["train_loss"], label=name)
plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("resnet_variants_comparison.png")
plt.show()

print("\nFinal Test Accuracy:")
for name in models.keys():
    print(f"{name}: {max(results[name]['test_acc']):.2f}%")