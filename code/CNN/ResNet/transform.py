from torchvision import transforms

def transform_cifar10(train):
	if train:
		transform = transforms.Compose([
				transforms.Pad(4),
				transforms.RandomHorizontalFlip(),
				transforms.RandomCrop(32),
				transforms.ToTensor()])
	else:
		transform = transforms.ToTensor()

	return transform
