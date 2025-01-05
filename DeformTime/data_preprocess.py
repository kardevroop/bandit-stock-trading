import pandas as pd
import numpy as np
import glob
import os

compile = None

master_df = None
for file in glob.glob(f'./data/dataset/SP500/stocks/*.csv'):
    filename = file.split("\\")[-1]

    if 'all' in filename:
        continue

    company = file.split("\\")[-1].split(".")[0]

    #if company in ["AKAM", "BXP", "CAG"]:
    print(file)

    dest = '/'.join(file.split("/")[:-1])

    if master_df is None:
        master_df = pd.read_csv(file)
        master_df = master_df.drop(columns=['TICKER', 'CUSIP', 'COMNAM'])
        master_df.set_index('date', inplace=True)
        master_df['SELL_PRC'] = master_df['PRC'] - master_df['TRAN_COST']
        columns = list(master_df.columns)
        n_columns = [f'{company}_{a}' for a in columns]
        #master_df = master_df.drop(columns=['date'])
        cm = {}
        for old, new in zip(columns, n_columns):
            cm[old] = new
        master_df = master_df.rename(columns=cm)


    else:
        df = pd.read_csv(file)
        df = df.drop(columns=['TICKER', 'CUSIP', 'COMNAM'])
        df['SELL_PRC'] = df['PRC'] - df['TRAN_COST']
        df.set_index('date', inplace=True)
        columns = list(df.columns)
        n_columns = [f'{company}_{a}' for a in columns]
        cm = {}
        for old, new in zip(columns, n_columns):
            cm[old] = new
        #df = df.drop(columns=['date'])
        df = df.rename(columns=cm)
        master_df = pd.merge(master_df, df, left_index=True, right_index=True)

tcolumn = "PRC"
pattern = f".*_{tcolumn}"

targets = list(master_df.filter(regex=pattern).columns)
print(targets)

if tcolumn == 'PRC':
    tmp = []    
    for t in targets:
        if "SELL_PRC" in t:
            continue
        tmp.append(t)
    targets = [a for a in tmp]

columns = list(master_df.columns)
for t in targets:
    columns.remove(t)

expected_targets = ['_'.join([a.split('_')[0], 'expected', tcolumn]) for a in targets]
for next_ret, curr_ret  in zip(expected_targets, targets):
    master_df[next_ret] = master_df[curr_ret].shift(-1)

targets2 = ['NDSN', 'HOLX', 'ATO', 'KIM', 'NVR', 'DPZ', 'JBHT', 'DECK', 'FRT', 'KMX', 'EXR', 'LNT', 'CINF', 'ED', 'REG', 'CPB', 'LH', 'TRMB', 'DVA', 'PKG', 'CAG', 'NTRS', 'KMB', 'TRV', 'RHI', 'UHS', 'EMN', 'PODD', 'TECH', 'EXPD', 'WRB', 'EIX', 'STLD', 'BXP', 'CHRW', 'IVZ', 'HSIC', 'TFX', 'AKAM', 'JKHY', 'HBAN', 'ESS', 'ETR', 'FFIV', 'CPT', 'IEX', 'IRM', 'COO', 'MHK', 'FDS']

predicted_targets = ['_'.join([a, 'predicted', tcolumn]) for a in targets2]
for next_ret  in predicted_targets:
    master_df[next_ret] = 0

# test_np = np.load('results\SP500a_1_DeformTime_SP500a_ftMS_sl336_ll0_pl1_dm32_nh4_el2_d0.0_ld0.0_lr0.001_\'Exp\'_0\pred.npy')
# test_np = test_np[:,0,:]
# print(test_np.shape)
# test_pred = pd.DataFrame(test_np, columns=predicted_targets)
# n = test_pred.shape[0]

# # print(test_pred.tail())

# # print(master_df.columns)

# for column in predicted_targets:
#     master_df[column].iloc[-n:] = test_pred[column].shift(-1)

master_df = master_df[columns + expected_targets + predicted_targets + targets]
# print(master_df.head())
# print(master_df.tail())

print(master_df.shape)
print(master_df.isnull().any().any())
# master_df = master_df.dropna()
print(master_df.shape)

if compile is None:
    compile = master_df
else:
    compile = pd.concat([compile, master_df])
    print(compile)
print(compile.shape)

    # master_df.to_csv(os.path.join(dest, 'all.csv'), index=False)
compile.insert(loc=0, column='date', value=compile.index)
compile.to_csv('./data/dataset/SP500/stocks/all_PRC.csv', index=False)