import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    x = np.abs(pred - true)
    print("diff shape: ", x.shape)
    x = np.mean(x, axis=0)
    print("mae along axis: ", x.shape)
    x = np.mean(x)
    print("mae for n_features: ", x)
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    x = (pred - true) ** 2
    print("diff shape: ", x.shape)
    x = np.mean(x, axis=0)
    print("mse along axis: ", x.shape)
    x = np.mean(x)
    print("mse for n_features: ", x)
    return np.mean((pred - true) ** 2)

#def MAE(pred, true):
#    return np.mean(np.abs(pred - true))


#def MSE(pred, true):
#    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def SMAPE(pred, true) -> float:
    return np.mean(
            np.abs(pred - true) / 
            ((np.abs(pred) + np.abs(true))/2)
        )*100


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
