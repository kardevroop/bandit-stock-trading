import os, glob
import pandas as pd 
import shutil
import numpy as np
import matplotlib.pyplot as plt


src = './outs'
dest = './processed'
returns = []
parameters = []

for file in glob.glob("tout_max_Extra_*.txt"):
    print(file)

    text = None
    with open(file, "r") as f:
        text = f.readlines()

    for line in text:
        if "Run" in line:
            returns.append(line[:-1])
            parameters.append(line[:-1])
        elif 'Starting with' in line:
            parameters.append(line[:-1])
        elif 'Return'in line:
            returns.append(line[:-1])


print(f"len of returns: {len(returns)}")
print(f"len of parameters: {len(parameters)}")
header = ['Return']
final = []
pool = {}
all_returns = {}
for i in range(0, len(parameters), 395):
    run = int(parameters[i].split(" ")[-1])
    running_pool = []
    running_returns = []
    for j in range(1, 395):
        parameter = float(parameters[i+j].split(" ")[-1])
        return_ = float(returns[i+j].split(" ")[-1])
        running_pool.append(parameter)
        running_returns.append(return_)

    pool[run] = running_pool
    all_returns[run] = running_returns


fig, ax = plt.subplots(5, 2)

for r in range(1, 6):

    x = list(range(len(pool[1])))
    y1 = pool[r]
    y2 = all_returns[r]

    ax[r-1, 0].plot(x, y1, label ='Investment')
    ax[r-1, 1].plot(x, y2, label ='Return', color='orange')
    ax[r-1, 0].legend()
    ax[r-1, 1].legend()

fig.suptitle('Daily Capital and returns of Stock Max Loss')
plt.savefig('stock_max_Extra_capital_return.png')
# plt.show()