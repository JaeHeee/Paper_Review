import torch
import torch.nn as nn
from dataset import *

torch.manual_seed(2020)


def conv3x3(in_channels, out_channels, stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, downsample=None):
		super(ResidualBlock, self).__init__()
		self.conv1 = conv3x3(in_channels, out_channels, stride)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU()
		self.conv2 = conv3x3(out_channels, out_channels)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.downsample = downsample

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		if self.downsample:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)
		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	def __init__(self, block, layers, num_classes=10):
		super(ResNet, self).__init__()
		self.in_channels = 16
		self.conv = conv3x3(3, 16)
		self.bn = nn.BatchNorm2d(16)
		self.relu = nn.ReLU(inplace=True)
		self.layer1 = self.make_layer(block, 16, layers[0])
		self.layer2 = self.make_layer(block, 32, layers[1], 2)
		self.layer3 = self.make_layer(block, 64, layers[2], 2)
		self.avg_pool = nn.AvgPool2d(8)
		self.fc = nn.Linear(64, num_classes)


	def make_layer(self, block, out_channels, blocks, stride=1):
		downsample = None
		
		if (stride != 1) or (self.in_channels != out_channels):
			downsample = nn.Sequential(
					conv3x3(self.in_channels, out_channels, stride=stride),
					nn.BatchNorm2d(out_channels))

		layers = []
		layers.append(block(self.in_channels, out_channels, stride, downsample))
		self.in_channels = out_channels
		
		for i in range(1, blocks):
			layers.append(block(out_channels, out_channels))
		return nn.Sequential(*layers)


	def forward(self, x):
		out = self.conv(x)
		out = self.bn(out)
		out = self.relu(out)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.avg_pool(out)
		out = out.view(out.size(0), -1)
		out = self.fc(out)

		return out


class Solver(object):
	def __init__(self, args):
		self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
		self.train_dataset = DatasetCIFAR10(True)
		self.test_dataset = DatasetCIFAR10(False)
		self.batch_size = args.batch_size
		self.learning_rate = args.learning_rate
		self.num_epochs = args.num_epochs
	
	def update_lr(self, optimizer, lr):
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr


	def solve(self):
		train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset.__getitem(), batch_size=self.batch_size, shuffle=True)
		test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset.__getitem(), batch_size=self.batch_size, shuffle=False)

		model = ResNet(ResidualBlock, [2, 2, 2]).to(self.device)
		
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
		
		total_step = len(train_loader)
		curr_lr = self.learning_rate
		for epoch in range(self.num_epochs):
			for i, (images, labels) in enumerate(train_loader):
				images = images.to(self.device)
				labels = labels.to(self.device)

				outputs = model(images)
				loss = criterion(outputs, labels)
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if (i+1) % 100 == 0:
					print(f"Epochs [{epoch+1}/{self.num_epochs}], Step [{i+1}/{total_step} Loss: {loss.item():.4f}]")

			if (epoch+1) % 20 == 0:
				curr_lr /= 3
				update_lr(optimizer, curr_lr)

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

			print(f'Accuracy of the model on the test images: {100 * correct/total} %')
		
		torch.save(model.state_dict(), 'resnet.pt')
