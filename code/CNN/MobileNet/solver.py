import torch
import torch.nn as nn
from dataset import *

torch.manual_seed(2020)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseConvBlock, self).__init__()
        self.depconv = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False, groups=in_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depconv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = DepthwiseConvBlock(in_channels, in_channels, stride=stride) 
        self.pointwise = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class MobileNet(nn.Module):
    def __init__(self, num_classes=100):
        super(MobileNet, self).__init__()
        self.network = nn.Sequential(
                ConvBlock(3, 32, kernel_size=3, stride=2, padding=1),
        
                DepthwiseSeparableConv(32, 64, stride=1),
                DepthwiseSeparableConv(64, 128, stride=2),
                DepthwiseSeparableConv(128, 128, stride=1),
                DepthwiseSeparableConv(128, 256, stride=2),
                DepthwiseSeparableConv(256, 256, stride=1),
                DepthwiseSeparableConv(256, 512, stride=2),

                DepthwiseSeparableConv(512, 512, stride=1),
                DepthwiseSeparableConv(512, 512, stride=1),
                DepthwiseSeparableConv(512, 512, stride=1),
                DepthwiseSeparableConv(512, 512, stride=1),
                DepthwiseSeparableConv(512, 512, stride=1),

                DepthwiseSeparableConv(512, 1024, stride=2),
                DepthwiseSeparableConv(1024, 1024, stride=2)
        )

        self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.network(x)
        x = self.ap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



class Solver(object):
    def __init__(self, args):
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.train_dataset = DatasetCIFAR100(True)
        self.test_dataset = DatasetCIFAR100(False)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs

    def solve(self):
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset.__getitem__(), batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset.__getitem__(), batch_size=self.batch_size, shuffle=False)

        model = MobileNet().to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=self.learning_rate)

        total_step = len(train_loader)
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 ==0:
                    print(f'Epochs [{epoch+1}/{self.num_epochs}], Step [{i+1}/{total_step} Loss: {loss.item():.4f}]')

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Accuracy of the model on the test images: {100 * correct/total}%')

        torch.save(model.state_dict(), 'MobileNet.pt')
                
