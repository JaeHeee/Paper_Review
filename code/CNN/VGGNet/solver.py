import torch
import torch.nn as nn
from dataset import *

torch.manual_seed(2020)

def conv_layer(in_channels, out_channels, kernel_size, padding):
    layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size = pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(in_size, out_size):
    layer = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.BatchNorm1d(out_size),
            nn.ReLU()
            )
    return layer


class VGG16(nn.Module):
    def __init__(self, n_classes=100):
        super(VGG16, self).__init__()

        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        self.layer6 = vgg_fc_layer(7*7*512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        self.layer8 = nn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out


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

        model = VGG16().to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        for epoch in range(self.num_epochs):
            avg_loss=0
            cnt = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                _, outputs = model(images)
                loss = criterion(outputs, labels)
                avg_loss += loss.data
                cnt += 1
                if cnt%100 == 0:
                    print(f'[Epochs: {epoch}] loss: {loss.data:4f} avg_loss: {avg_loss/cnt:4f}')
                loss.backward()
                optimizer.step()
            scheduler.step(avg_loss)

        model.eval()
        correct = 0
        total = 0

        for idx, (images, labels) in enumerate(test_loader):
            images = images.to(self.device)
            _, outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()
            if idx%100 == 0:
                print(f'avg acc: {100* correct/total:2f}')
            print(f'avg acc: {100* correct/total:2f}')

        torch.save(model.state_dict(), 'vgg16.pt')






