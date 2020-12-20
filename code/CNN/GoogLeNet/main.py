import argparse
from solver import Solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('--num_epochs', default=50, type=int, help='epochs')

    args = parser.parse_args()
    solver = Solver(args)
    solver.solve()
