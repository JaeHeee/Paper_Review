import torch
from dataset import *

torch.manual_seed(2020)


class Solver(object):
	def __init__(self, args):
		self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu')
		self.train_dataset = DatasetCIFAR10(True)
		self.test_dataset = DatasetCIFAR10(False)
		self.batch_size = args.batch_size




	def solve():
		train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
		test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)


