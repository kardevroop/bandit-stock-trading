import os, glob
import pandas as pd 
import shutil
import numpy as np
import matplotlib.pyplot as plt


src = './outs'
dest = './processed'
stock = "DPZ"
returns = []
parameters = []
v_i_caps = []

for file in glob.glob("tout_noExtra_pred_336.txt"):
    print(file)

    text = None
    with open(file, "r") as f:
        text = f.readlines()

    for line in text:
        if "Run" in line:
            returns.append(line[:-1])
            parameters.append(line[:-1])
            v_i_caps.append(line[:-1])
        elif 'Starting with' in line:
            parameters.append(line[:-1])
        elif 'Return' in line:
            returns.append(line[:-1])
        elif f"{stock}" in line and 'v_i_cap' in line:
            v_i_caps.append(line[:-1])


print(f"len of returns: {len(returns)}")
print(f"len of parameters: {len(parameters)}")
print(f"len of v_i_caps: {len(v_i_caps)}")

header = ['Return']
final = []
pool = {}
all_returns = {}
all_vi_caps = {}
all_price = {}


for i in range(0, len(parameters), 250):
    run = int(parameters[i].split(" ")[-1])
    running_pool = []
    running_returns = []
    running_v = []
    running_prc = []

    for j in range(1, 250):
        parameter = float(parameters[i+j].split(" ")[-1])
        return_ = float(returns[i+j].split(" ")[-1])
        running_pool.append(parameter)
        running_returns.append(return_)

        v_i_list = v_i_caps[i+j].split(" ")
        idx = v_i_list.index('v_i_cap:') + 1
        v_i = float(v_i_list[idx])
        running_v.append(v_i)

        idx = v_i_list.index('PRC_t+1:') + 1
        v_i = float(v_i_list[idx])   
        running_prc.append(v_i)

    pool[run] = running_pool
    all_returns[run] = running_returns
    all_vi_caps[run] = running_v
    all_price[run] = running_prc


fig, ax = plt.subplots(5, 3, figsize=(20, 15))

ax[0, 0].set_title("Investment")
ax[0, 1].set_title("Return")
ax[0, 2].set_title("V_i_cap")
# ax[0, 3].set_title("PRC")

for r in range(1, 6):

    x = list(range(len(pool[1])))
    y1 = pool[r]
    y2 = all_returns[r]
    y3 = all_vi_caps[r]
    prc = all_price[r]

    ax[r-1, 0].set_ylabel(f"Run {r}")

    ax[r-1, 0].plot(x, y1, label ='Investment')
    ax[r-1, 1].plot(x, y2, label ='Return', color='orange')

    # ax[r-1, 2].plot(x, [100 for _ in range(len(x))], label ='Max', color='red', linestyle='--')
    ax[r-1, 2].plot(x, y3, label ='V_i', color='magenta')

    ax_prc = ax[r-1, 2].twinx().twiny()

    ax_prc.plot(x, prc, label ='price', color='red')
    ax_prc.legend()
    

fig.suptitle(f"Daily Capital and returns of Stock Loss (with V_i_caps for {stock})")
plt.savefig('./plots/trades/stock_noExtra_capital_return_vi.png')
# plt.show()