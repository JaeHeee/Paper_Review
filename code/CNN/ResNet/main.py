import argparse
from solver import Solver


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', default=128, type=int, help='batch size')
	parser.add_argument()
	parser.add_argument()



	args = parser.parse_args()
	solver = Solver(args)
	solver.solve()
