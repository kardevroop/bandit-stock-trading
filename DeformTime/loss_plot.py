import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.utils.losses import stock_loss, stock_loss_l2_norm, stock_loss_max_norm, stock_loss_global_norm
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def gradient_stock_loss(k, nn_output, target, target_next, loss = "Stock"):
    signs = nn_output / torch.abs(nn_output)

    s = torch.sum(torch.abs(nn_output))

    grad = torch.zeros_like(nn_output)

    grad[0,k] = (s - nn_output[0,k]*signs[0, k]) / s**2

    for j in range(grad.shape[1]):
        if j == k:
            continue
        
        grad[0,j] = - nn_output[0,j]*signs[0,k] / s**2

    return - torch.sum(torch.flatten(target_next - target) * torch.flatten(grad))




def main(args):
    num_stocks = args.stocks
    itr = args.points

    if args.loss == 'Stock':
        criterion = stock_loss(args.enable_action)
    elif args.loss == 'Stock_l2':
        criterion = stock_loss_l2_norm(args.enable_action)
    elif args.loss == 'Stock_max':
        criterion = stock_loss_max_norm(args.enable_action)
    elif args.loss == 'Stock_global':
        criterion = stock_loss_global_norm(args.enable_action)

    X=[]
    Y=[]
    grad_Y = []


    i = 6

    target_next = torch.ones(1, num_stocks)
    target = torch.zeros(1, num_stocks)

    for x in range(itr):
        O_i = torch.tanh(-5  + torch.rand(1, num_stocks) * 10)
        O_i = torch.tensor(O_i, requires_grad=True)
        print(O_i.shape)

        x = O_i[0,i]
        y = criterion(O_i, target, target_next=target_next)

        y.backward()
        print(O_i.grad.shape)

        print(O_i.grad)

        # x = x.numpy()
        # X.append(x)
        # y = y.detach().numpy()
        # Y.append(y)


        gradient_at_i = gradient_stock_loss(i, O_i, target, target_next, args.loss)
        grad_Y.append(gradient_at_i)
        

        # print(x)
        # print(y)


    # plt.scatter(X, Y, color='b', label='Stock Loss')
    # # plt.savefig('stock_loss.png')

    # plt.scatter(X, grad_Y, color='r', label='Gradient of Stock Loss')

    # plt.xlabel('o_i')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig('stock_loss_gradient.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MTS forecasting')

    parser.add_argument('--loss', type=str, default='Stock', help='loss function')
    parser.add_argument('--points', type=int, default='500', help='number of random points to generate')
    parser.add_argument('--stocks', type=int, default='50', help='number of output neurons')
    parser.add_argument('--enable_action', action='store_true', help="Set flag to enable extra output node to do nothing")

    args = parser.parse_args()

    main(args)