import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.utils.losses import stock_loss, stock_loss_l2_norm, stock_loss_max_norm, stock_loss_global_norm, soft_stock_loss
import argparse
import shutil
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def gradient_stock_loss(k, nn_output, target, target_next):
    signs = nn_output / torch.abs(nn_output)

    s = torch.sum(torch.abs(nn_output))

    grad = torch.zeros_like(nn_output)

    grad[0,k] = (s - nn_output[0,k]*signs[0, k]) / s**2

    for j in range(grad.shape[1]):
        if j == k:
            continue
        
        grad[0,j] = - nn_output[0,j]*signs[0,k] / s**2

    return - torch.sum(torch.flatten(target_next - target) * torch.flatten(grad))

def gradient_soft_stock_loss(k, nn_output, target, target_next, gamma=5):

    signs = torch.tanh(gamma * nn_output)

    v_i_caps = torch.abs(nn_output) / torch.sum(torch.abs(nn_output))

    s = torch.sum(torch.abs(nn_output))

    grad = torch.zeros_like(nn_output)

    grad[0,k] = v_i_caps[0, k] * (1 - torch.tanh(gamma * nn_output[0,k])*torch.tanh(gamma * nn_output[0,k])) * gamma \
        + torch.tanh(gamma * nn_output[0,k]) * (s*signs[0, k] - nn_output[0,k]) / s**2

    for j in range(grad.shape[1]):
        if j == k:
            continue
        
        grad[0,j] = - torch.tanh(gamma * nn_output[0,j]) * nn_output[0,j]*signs[0,k] / s**2

    return - torch.sum(torch.flatten(target_next - target) * torch.flatten(grad))

def gradient_stock_loss_max_norm(k, nn_output, target, target_next):
    signs = nn_output / torch.abs(nn_output)

    s = torch.sum(torch.abs(nn_output))

    grad = torch.zeros_like(nn_output)

    max_diff = torch.max(torch.abs(target_next - target), 2)[0]
    # print(f"max_diff: {max_diff[0,0]}")

    # print(target.shape)

    a = torch.flatten(target_next - target)
    # print(a)
    a = torch.reshape(a, target.shape)
    b, seq, f = target.shape[0], target.shape[1], target.shape[2]
    a[0,0,:] = torch.div(a[0,0,:], max_diff)
    # print(a)

    grad[0,k] = (s - nn_output[0,k]*signs[0, k]) / s**2

    for j in range(grad.shape[1]):
        if j == k:
            continue
        
        grad[0,j] = - nn_output[0,j]*signs[0,k] / s**2

    return - torch.sum(torch.flatten(a) * torch.flatten(grad))


def gradient_stock_loss_l2_norm(k, nn_output, target, target_next):
    signs = nn_output / torch.abs(nn_output)

    s = torch.sum(torch.abs(nn_output))

    grad = torch.zeros_like(nn_output)

    max_diff = torch.max(torch.abs(target_next - target), 2)[0]
    # print(f"max_diff: {max_diff[0,0]}")

    # print(target.shape)

    v_i_cap = torch.abs(nn_output) / torch.sum(torch.abs(nn_output)).item()

    a = torch.flatten(target_next - target)
    # print(a)
    a = torch.reshape(a, target.shape)
    b, seq, f = target.shape[0], target.shape[1], target.shape[2]
    a[0,0,:] = torch.div(a[0,0,:], max_diff)
    # print(a)

    prod = 1.0 / (2.0 * torch.sqrt(torch.sum(torch.flatten(v_i_cap) * torch.flatten(a**2))))

    grad[0,k] = (s*signs[0,k] - torch.abs(nn_output[0,k])*signs[0, k]) / s**2

    for j in range(grad.shape[1]):
        if j == k:
            continue
        
        grad[0,j] = - torch.abs(nn_output[0,j])*signs[0,k] / s**2

    return - prod * torch.sum(torch.flatten(a) * torch.flatten(grad))

def gradient_stock_loss_global_norm(k, nn_output, target, target_next):
    signs = nn_output / torch.abs(nn_output)

    s = torch.dot(torch.flatten(torch.abs(nn_output)), torch.flatten((target_next - target)))

    print(target.shape)

    grad = torch.zeros_like(nn_output)

    grad[0,k] = (s - nn_output[0,k] * (target_next - target)[0, 0, k] * signs[0, k]) / s**2

    for j in range(grad.shape[1]):
        if j == k:
            continue
        
        grad[0,j] = - nn_output[0,j] * (target_next - target)[0, 0, k] * signs[0,k] / s**2

    return - torch.sum(torch.flatten(target_next - target) * torch.flatten(grad))


def gradient_loss(k, nn_output, target, target_next, loss_func = "Stock"):
    if loss_func == 'Stock':
        grad = gradient_stock_loss(k, nn_output, target, target_next)
    elif loss_func == 'Stock_l2':
        grad = gradient_stock_loss_l2_norm(k, nn_output, target, target_next)
    elif loss_func == 'Stock_max':
        grad = gradient_stock_loss_max_norm(k, nn_output, target, target_next)
    elif loss_func == 'Stock_global':
        grad = gradient_stock_loss_global_norm(k, nn_output, target, target_next)
    elif loss_func == 'Stock_soft':
        grad = gradient_soft_stock_loss(k, nn_output, target, target_next, gamma=100)

    return grad


def main(args):
    num_stocks = args.stocks

    if args.loss == 'Stock':
        criterion = stock_loss(args.enable_action)
    elif args.loss == 'Stock_l2':
        criterion = stock_loss_l2_norm(args.enable_action)
    elif args.loss == 'Stock_max':
        criterion = stock_loss_max_norm(args.enable_action)
    elif args.loss == 'Stock_global':
        criterion = stock_loss_global_norm(args.enable_action)
    elif args.loss == 'Stock_soft':
        criterion = soft_stock_loss(args.enable_action)


    # i = 0

    if args.mode == 'long':
        target_next = torch.abs(0.5 * torch.ones(1, num_stocks))
        target = torch.zeros(1, num_stocks)
    else:
        target_next = torch.zeros(1, num_stocks)
        target = torch.abs(0.5 * torch.ones(1, num_stocks))

    # O_i = torch.ones(1, num_stocks+1) if args.enable_action else torch.ones(1, num_stocks)

    O_i_1 = torch.tanh(-5 + 10 * torch.rand(1, num_stocks+1)) if args.enable_action else torch.tanh(-5 + 10 * torch.rand(1, num_stocks))
        
    for i in range(num_stocks):
        print(f"\n\nOutput O_{i+1}")
        O_i = torch.clone(O_i_1)

        X=[]
        Y=[]
        grad_Y = []

        for x in np.linspace(-1.0, 1.0, num=args.points):

            # O_i = x * O_i
            
            # O_i = torch.tensor(O_i)
            # print(O_i)

            # if args.enable_action:
            #     O_i[0, num_stocks] = torch.tanh(-5 + 10 * torch.rand(1)[0])

            O_i[0,i] = x

            # print(f"O_i: {O_i}")

            xx = O_i[0,i]


            if len(target.shape) < 3:
                target = target.unsqueeze(0)
                target_next = target_next.unsqueeze(0)

            y = criterion(O_i, target, target_next=target_next)
            # print(f"y: {y}")

            # y.backward()
            # print(O_i.grad.shape)

            # print(O_i.grad)

            xp = xx.clone().detach().numpy()
            X.append(xp)
            yp = y.clone().detach().numpy()
            Y.append(yp)

            grad = torch.zeros_like(O_i)
            for j in range(grad.shape[1]):
                gradient_at_j = gradient_loss(j, O_i, target, target_next, loss_func=args.loss)
                grad[0,j] = gradient_at_j

            # print(grad)

            grad_Y.append(grad[0,i])
            # grad_Y.append(O_i.grad[0,i].numpy())
            

            # print(x)
            # print(y)

        comp = '>' if args.mode == 'long' else '<'

        fig, ax1 = plt.subplots()

        ax1.set_xlabel(f'NN output: o_{i+1}')
        ax1.set_ylabel('Loss')
        fig.suptitle(f"{args.loss} Loss function and gradient Values for Ret_t+1 {comp} Ret_t {'with Hold' if args.enable_action else ''}")

        ax1.scatter(X, Y, color='b', label='Stock Loss')

        ax2 = ax1.twinx()

        ax2.set_ylabel('Gradient')
        ax2.scatter(X, grad_Y, color='r', label='Gradient of Stock Loss')

        fig.tight_layout()

        out_path = os.path.join("plots", "loss", f"{args.loss}", f"st{args.stocks}", f"o{i+1}")
        if not os.path.exists(out_path):
            # shutil.rmtree(out_path)
            os.makedirs(out_path)
    

        plt.savefig(os.path.join(out_path, f"{args.loss}_loss_gradient_st{args.stocks}_{args.mode}{'_h' if args.enable_action else ''}.png"))
        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loss Function')

    parser.add_argument('--loss', type=str, default='Stock', help='loss function')
    parser.add_argument('--points', type=int, default='500', help='number of random points to generate')
    parser.add_argument('--stocks', type=int, default='50', help='number of output neurons')
    parser.add_argument('--enable_action', action='store_true', help="Set flag to enable extra output node to do nothing")
    parser.add_argument('--mode', type=str, choices=['long', 'short'], help="Set mode for Ret_t+1 > or < Ret_t")

    args = parser.parse_args()

    main(args)