import torch
import torch.nn as nn
from dataset import *

torch.manual_seed(2020)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, filter_list):
        super(InceptionModule, self).__init__()

        branch1_filters, branch2_1_filters, branch2_2_filters, branch3_1_filters, branch3_2_filters, branch4_2_filters = filter_list
        self.branch1_filters = branch1_filters
        self.branch2_1_filters = branch2_1_filters       
        self.branch2_2_filters = branch2_2_filters
        self.branch3_1_filters = branch3_1_filters
        self.branch3_2_filters = branch3_2_filters
        self.branch4_2_filters = branch4_2_filters

        self.branch1 = nn.Sequential(
                ConvBlock(in_channels, self.branch1_filters, kernel_size=1, stride=1, padding=0)
        )
        
        self.branch2 = nn.Sequential(
                ConvBlock(in_channels, self.branch2_1_filters, kernel_size=1, stride=1, padding=0),
                ConvBlock(self.branch2_1_filters, self.branch2_2_filters, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
                ConvBlock(in_channels, self.branch3_1_filters, kernel_size=1, stride=1, padding=0),
                ConvBlock(self.branch3_1_filters, self.branch3_2_filters, kernel_size=5, stride=1, padding=2)
        )

        self.branch4 = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
                ConvBlock(in_channels, self.branch4_2_filters, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.AP = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.AP(x)
        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=100):
        super(GoogLeNet, self).__init__()

        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        self.conv2 = ConvBlock(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        
        self.inception3a = InceptionModule(192, [64, 96, 128, 16, 32, 32])
        self.inception3b = InceptionModule(256, [128, 128, 192, 32, 96, 64])
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.inception4a = InceptionModule(480, [192, 96, 208, 16, 48, 64])
        self.aux4a = AuxiliaryClassifier(512, num_classes)
        self.inception4b = InceptionModule(512, [160, 112, 224, 24, 64, 64])
        self.inception4c = InceptionModule(512, [128, 128 ,256, 24, 64, 64])
        self.inception4d = InceptionModule(512, [112, 144, 288, 32, 64, 64])
        self.aux4d = AuxiliaryClassifier(528, num_classes)
        self.inception4e = InceptionModule(528, [256, 160, 320, 32, 128, 128])
        self.pool4 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.inception5a = InceptionModule(832, [256, 160, 320, 32, 128, 128])
        self.inception5b = InceptionModule(832, [384, 192, 384, 48, 128, 128])
        
        self.AP = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        
        x = self.inception4a(x)
        aux4a = self.aux4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux4d = self.aux4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.AP(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x, aux4a, aux4d


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

        model = GoogLeNet().to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        total_step = len(train_loader)
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs, aux4a, aux4d  = model(images)
                output_loss = outputs + aux4a*0.3 + aux4d*0.3
                loss = criterion(output_loss, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print(f'Epochs [{epoch+1}/{self.num_epochs}], Step [{i+1}/{total_step} Loss: {loss.item():.4f}]')

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs, _, _ = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Accuracy of the model on the test images: {100 * correct/total}%')

        torch.save(model.state_dict(), 'GoogLeNet.pt')






