import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data

import torchvision.transforms as transforms

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.ConvReLUBN = nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        conv3x3(out_planes, out_planes),
        nn.BatchNorm2d(out_planes))
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x
        out = self.ConvReLUBN(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.Conv2d(out_planes, out_planes*4, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_planes*4))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes = 200):
        super(ResNet50, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample=None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    device = torch.device(0)
    net = ResNet50(Bottleneck, [3,4,6,3])
    net.to(device)
    print('Usage gpu {0}'.format(0))

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    # load data set
    print('Reading data...')
    train_dir = '../data/tiny-imagenet/tiny-imagenet-200/train'
    train_dataset = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
    train_loader = data.DataLoader(train_dataset, batch_size=32)
    print('Load: {}'.format(train_dir))

    # train model
    for epoch in range(10):
        print('-EPOCH: {}'.format(epoch))
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            input, target = data

            input = torch.tensor(input).to(device)
            target = torch.tensor(target).to(device)

            output = net(input)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 50 == 49:
                print('Train:{0}, Loss: {1}'.format(i+1, running_loss / 50))
                running_loss = 0.0
    print('training complete')
    torch.save(net.state_dict(), '../save_model/baseline-resnet50.pt')







