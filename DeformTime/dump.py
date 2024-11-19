import os, glob
import pandas as pd 
import shutil
import numpy as np


src = './outs'
dest = './processed'
metrics = []
parameters = []

for file in glob.glob(os.path.join(src, "sp500", "*.o")):
    print(file)
    dataset = file.split("/")[2]
    print(dataset)
    text = None
    with open(file, "r") as f:
        text = f.readlines()
    #text = text.split("\n")
    #print(text)

    for line in text:
        if "Processing" in line:
            metrics.append(line)
            parameters.append(line)
        elif 'parameters:' in line:
            parameters.append(line)
        elif 'target mae:' in line:
            metrics.append(line)
        elif 'target mse:' in line:
            metrics.append(line)
    #print(metrics)
    # metrics = [[float(a.split(", ")[1][4:-3]), float(a.split(", ")[0][4:])] for a in metrics]
    # #print(metrics)
    # seq_1, seq_6, seq_12, seq_24 = metrics[:10], metrics[10:20], metrics[20:30], metrics[30:40]
    # df = [pd.DataFrame(seq_1, columns = ['MAE', 'MSE'])
    #                             ,pd.DataFrame(seq_6, columns = ['MAE', 'MSE'])
    #                             ,pd.DataFrame(seq_12, columns = ['MAE', 'MSE'])
    #                             ,pd.DataFrame(seq_24, columns = ['MAE', 'MSE'])]
    # directories = [os.path.join(dest, dataset, seq_len) for seq_len in ['1', '6', '12', '24']]
    # for i in range(len(directories)):
    #     directory = directories[i]
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     df[i].to_csv(os.path.join(directory, "auto.csv"), index=False)
print(metrics)
print(parameters)
header = ['Company', 'Parameters', 'Best MAE', 'Average MAE', 'Best MSE', 'Average MSE']
final = []
param = {}
for i in range(0, len(parameters), 6):
    company = parameters[i].split(" ")[-2]
    parameter = parameters[i+1].split(" ")[-1][:-1]
    print(company)
    param[company] = parameter
for i in range(0, len(metrics), 11):
    company = metrics[i].split(" ")[-2]
    print(company)
    m = metrics[i+1:i+11]
    mae = []
    mse = []
    for j in range(10):
        #print(metrics[i+j+1])
        if j%2==0:
            mae.append(float(metrics[i+j+1].split(":")[1][:-1]))
        else:
            mse.append(float(metrics[i+j+1].split(":")[1][:-1]))
    #print(mae)
    #print(mse)
    avg_mae, avg_mse = np.average(mae), np.average(mse)
    best_mae, best_mse = np.min(mae), np.min(mse)
    final.append([company, param[company], best_mae, avg_mae, best_mse, avg_mse])
final.sort(key=lambda x: x[0])
final = pd.DataFrame(final, columns=header)
print(final.head())
final.to_csv(os.path.join(dest, 'sp500.csv'), index=False)