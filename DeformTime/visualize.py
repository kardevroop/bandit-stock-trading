import os, glob
import pandas as pd 
import shutil
import numpy as np
import matplotlib.pyplot as plt
import ast


src = './outs'
dest = './processed'
stock = "WRB"
returns = []
parameters = []
v_i_caps = {}
nn_outputs = {}
st_returns = {}

companies = ['AKAM', 'ATO', 'BXP', 'CAG', 'CHRW', 'CINF', 'COO', 'CPB', 'CPT', 'DECK', 'DPZ', 'DVA', 'ED', 'EIX', 'EMN', 'ESS', 'ETR', 'EXPD', 'EXR', 'FDS', 'FFIV', 'FRT', 'HBAN', 'HOLX', 'HSIC', 'IEX', 'IRM', 'IVZ', 'JBHT', 'JKHY', 'KIM', 'KMB', 'KMX', 'LH', 'LNT', 'MHK', 'NDSN', 'NTRS', 'NVR', 'PKG', 'PODD', 'REG', 'RHI', 'STLD', 'TECH', 'TFX', 'TRMB', 'TRV', 'UHS', 'WRB']

for file in glob.glob("tout_noExtra_pred_336.txt"):
    print(file)

    text = None
    with open(file, "r") as f:
        text = f.readlines()

    for line in text:
        if "Run" in line:
            returns.append(line[:-1])
            parameters.append(line[:-1])
            # v_i_caps.append(line[:-1])
        elif 'Starting with' in line:
            parameters.append(line[:-1])
        elif 'Return' in line:
            returns.append(line[:-1])
        elif 'using v_i_cap' in line:
            tokens = line.split(" ")
            idx = tokens.index('stock') + 1
            st = tokens[idx]
            if st not in v_i_caps.keys():
                v_i_caps[st] = []
            v_i_caps[st].append(line[:-1])
        elif 'nn_output:' in line:
            o_i = ast.literal_eval(line.split(":")[-1].strip())
            for st, o in zip(companies, o_i):
                if st not in nn_outputs.keys():
                    nn_outputs[st] = []
                nn_outputs[st].append(o)
        elif 'reward:' in line and 'go' in line:
            tokens = line.split(" ")
            idx = tokens.index('stock') + 1
            st = tokens[idx]
            if st not in st_returns.keys():
                st_returns[st] = []
            st_returns[st].append(tokens[-1])



print(f"len of returns: {len(returns)}")
print(f"len of parameters: {len(parameters)}")

ks = list(v_i_caps.keys())
print(f"len of v_i_caps: {len(v_i_caps[ks[0]])}")

header = ['Return']
final = []
pool = {}
all_returns = {}
all_vi_caps = {}
all_price = {}
all_nn_output = {}
all_st_returns = {}


for i in range(0, len(parameters), 250):
    run = int(parameters[i].split(" ")[-1])
    running_pool = []
    running_returns = []
    # running_v = []
    # running_prc = []

    for j in range(1, 250):
        parameter = float(parameters[i+j].split(" ")[-1])
        return_ = float(returns[i+j].split(" ")[-1])
        running_pool.append(parameter)
        running_returns.append(return_)

        # v_i_list = v_i_caps[i+j].split(" ")
        # idx = v_i_list.index('v_i_cap:') + 1
        # v_i = float(v_i_list[idx])
        # running_v.append(v_i)

        # idx = v_i_list.index('PRC_t+1:') + 1
        # v_i = float(v_i_list[idx])   
        # running_prc.append(v_i)
    
    pool[run] = running_pool
    all_returns[run] = running_returns

    for stock in v_i_caps.keys():
        running_v = []
        running_prc = []
        running_nn = []
        running_ret = []

        j = (run - 1) * 249
        for k in range(249):
            v_i_list = v_i_caps[stock][j+k].split(" ")

            idx = v_i_list.index('v_i_cap:') + 1
            v_i = float(v_i_list[idx])
            running_v.append(v_i)

            idx = v_i_list.index('PRC_t+1:') + 1
            v_i = float(v_i_list[idx])   
            running_prc.append(v_i)

            nn_value = nn_outputs[stock][j+k]
            running_nn.append(float(nn_value))

            stock_ret = st_returns[stock][j+k]
            running_ret.append(float(stock_ret))

        if run not in all_vi_caps.keys():
            all_vi_caps[run] = {}
        elif stock not in all_vi_caps[run].keys():
            all_vi_caps[run][stock] = None
        all_vi_caps[run][stock] = running_v

        if run not in all_price.keys():
            all_price[run] = {}
        elif stock not in all_price[run].keys():
            all_price[run][stock] = None
        all_price[run][stock] = running_prc

        if run not in all_nn_output.keys():
            all_nn_output[run] = {}
        elif stock not in all_nn_output[run].keys():
            all_nn_output[run][stock] = None
        all_nn_output[run][stock] = running_nn

        if run not in all_st_returns.keys():
            all_st_returns[run] = {}
        elif stock not in all_st_returns[run].keys():
            all_st_returns[run][stock] = None
        all_st_returns[run][stock] = running_ret


# ax[0, 0].set_title("Investment")
# ax[0, 1].set_title("Return")
# ax[0, 2].set_title("V_i_cap")
# ax[0, 3].set_title("PRC")

for r in range(1, 6):
    fig, ax = plt.subplots(1, 1, figsize=(40, 20))
    plt.rcParams.update({'font.size': 22})

    x = list(range(len(pool[1])))
    y1 = pool[r]
    y2 = all_returns[r]


    ax.plot(x, y1, label ='Investment')
    ax.set_yticklabels([int(a) for a in y1], fontsize=25)

    ax2 = ax.twinx().twiny()  # instantiate a second Axes that shares the same x-axis
    ax2.plot(x, y2, color='orange', label='Return')
    ax2.legend()

    ax.set_ylabel('Daily Money', fontsize=25)
    ax2.set_ylabel('Daily Return')

    ax.set_title('Daily Money vs Daily Return')

    if not os.path.exists(f"./plots/trades/returns/{r}"):
        os.makedirs(f"./plots/trades/returns/{r}")

    plt.savefig(f"./plots/trades/returns/{r}/stock_noExtra.png")
    plt.close()

for r in range(1, 6):
    x = list(range(len(pool[1])))

    # ax[r-1, 0].set_ylabel(f"Run {r}")
    for i, st in enumerate(companies):
        fig, ax = plt.subplots(1, 2, figsize=(50, 15))
        plt.rcParams.update({'font.size': 35})
        fig.suptitle("V_i_cap variation on Stock Loss")

        y2 = all_st_returns[r][st]
        y3 = all_vi_caps[r][st]
        # ret = all_returns[r][st]
        nn = all_nn_output[r][st]

        # ax.plot(x, y2, color='orange', linestyle = '--', label='Net Return')
        ax[0].plot(x, [a/100 for a in y3])
        # ax[0,0].set_title(f"Run {r}: V_i_cap for {st}")
        ax[0].set_ylabel('V_i_cap')
        ax[0].set_xlabel('timestep')
        # ax[0].legend()

        # ax[0].set_yticklabels([a/100 for a in y3], fontsize=22)

        ax2 = ax[0].twinx().twiny()  # instantiate a second Axes that shares the same x-axis
        ax2.plot(x, nn, color='magenta', label='NN Output')
        ax2.legend()

        ax[0].set_title("V_i_cap vs O_i ( or V_i )")

        ax[1].plot(x, [a/100 for a in y3])
        # ax[0,1].set_title(f"Run {r}: V_i_cap for {st}")
        ax[1].set_ylabel('V_i_cap')
        ax[1].set_xlabel('timestep')
        # ax[1].legend()

        ax3 = ax[1].twinx().twiny()  # instantiate a second Axes that shares the same x-axis
        ax3.plot(x, y2, color='red', label='Return')

        ax3.legend()

        ax[1].set_title("V_i_cap vs Ret for stock")

        if not os.path.exists(f"./plots/trades/v_i_cap/{r}/{st}"):
            os.makedirs(f"./plots/trades/v_i_cap/{r}/{st}")

        plt.savefig(f"./plots/trades/v_i_cap/{r}/{st}/stock_noExtra_vi.png")
        plt.close()



    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    for i, st in enumerate(companies):
        y3 = all_vi_caps[r][st]
        
        ax.plot(x, y3, label = f"{st}")
        ax.set_title(f"Run {r}: V_i_cap for all stocks")
        ax.set_ylabel('V_i')
        ax.set_xlabel('timestep')
        ax.legend()

        if not os.path.exists(f"./plots/trades/v_i_cap/{r}"):
            os.makedirs(f"./plots/trades/v_i_cap/{r}")

        plt.rcParams.update({'font.size': 22})
        plt.savefig(f"./plots/trades/v_i_cap/{r}/stock_noExtra_vi.png")

    plt.close()

# fig.suptitle(f"Daily Capital and returns of Stock Max Loss (with V_i_caps for {stock})")

# fig.suptitle("V_i_cap variation on Stock Loss")
# plt.savefig('./plots/trades/v_i_cap/stock_noExtra_vi.png')
# plt.show()