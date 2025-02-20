# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)

class stock_loss(nn.Module):
    def __init__(self, extra_node=False):
        super(stock_loss, self).__init__()
        self.extra_node = extra_node

    def forward(self, nn_output, target, **kwargs):

        signs = nn_output / t.abs(nn_output)

        s = t.sum(t.abs(nn_output), (1, 2))

        v_i_caps = t.zeros_like(nn_output)
        for b in range(v_i_caps.shape[0]):
            v_i_caps[b,:,:] = t.abs(nn_output[b,:,:]) / s[b]

        assert t.abs(t.sum(v_i_caps) - v_i_caps.shape[0]) < 0.001
        assert t.all(v_i_caps >= 0)


        if "target_next" not in kwargs:
            target_next = t.zeros_like(target)
        else:
            target_next = kwargs["target_next"]

        target = t.abs(target)
        target_next = t.abs(target_next)

        dim = v_i_caps.dim()

        # print(f"target: {target}")
        # print(f"target next: {target_next}")

        if self.extra_node:
            if dim == 2:
                hold_cap = v_i_caps[:,-1]
                v_i_caps = v_i_caps[:,:-1]
                signs = signs[:,:-1]
                # target = target[:,:-1]
                # target_next = target_next[:,:-1]
            else:
                hold_cap = v_i_caps[:,:,-1]
                v_i_caps = v_i_caps[:,:,:-1]
                signs = signs[:,:,:-1]
                # target = target[:,:,:-1]
                # target_next = target_next[:,:,:-1]
        else:
            hold_cap = t.zeros_like(v_i_caps)
        hold_cap = hold_cap.unsqueeze(1)

        # print(f"[INFO   ]       target shape: {target.shape}")
        # print(f"[INFO   ]       target next shape: {target_next.shape}")
        # print(f"[INFO   ]       weight shape: {weights.shape}")
        # print(f"[INFO   ]       hold cap shape: {hold_cap.shape}")
        # print(f"[INFO   ]       sign cap shape: {signs.shape}")
        # print(f"[INFO   ]       sum shape: {t.dot(t.flatten(weights), t.flatten(t.abs(target_next - target))).shape}")
        
        return -1 * ((t.dot(t.flatten(v_i_caps), t.flatten(target_next - target) * t.flatten(signs)) + t.sum(hold_cap)))

class soft_stock_loss(nn.Module):
    def __init__(self, extra_node=False):
        super(soft_stock_loss, self).__init__()
        self.extra_node = extra_node

    def forward(self, nn_output, target, **kwargs):

        if "gamma" not in kwargs:
            gamma = 5
        else:
            gamma = kwargs["gamma"]

        signs = t.tanh(gamma * nn_output)

        s = t.sum(t.abs(nn_output), (1, 2))

        v_i_caps = t.zeros_like(nn_output)
        for b in range(v_i_caps.shape[0]):
            v_i_caps[b,:,:] = t.abs(nn_output[b,:,:]) / s[b]

        assert t.abs(t.sum(v_i_caps) - v_i_caps.shape[0]) < 0.001
        assert t.all(v_i_caps >= 0)

        if "target_next" not in kwargs:
            target_next = t.zeros_like(target)
        else:
            target_next = kwargs["target_next"]

        target = t.abs(target)
        target_next = t.abs(target_next)

        dim = v_i_caps.dim()

        if self.extra_node:
            if dim == 2:
                hold_cap = v_i_caps[:,-1]
                v_i_caps = v_i_caps[:,:-1]
                signs = signs[:,:-1]
                # target = target[:,:-1]
                # target_next = target_next[:,:-1]
            else:
                hold_cap = v_i_caps[:,:,-1]
                v_i_caps = v_i_caps[:,:,:-1]
                signs = signs[:,:,:-1]
                # target = target[:,:,:-1]
                # target_next = target_next[:,:,:-1]
        else:
            hold_cap = t.zeros_like(v_i_caps)
        hold_cap = hold_cap.unsqueeze(1)

        # print(f"[INFO   ]       target shape: {target.shape}")
        # print(f"[INFO   ]       target next shape: {target_next.shape}")
        # print(f"[INFO   ]       weight shape: {weights.shape}")
        # print(f"[INFO   ]       hold cap shape: {hold_cap.shape}")
        # print(f"[INFO   ]       sign cap shape: {signs.shape}")
        # print(f"[INFO   ]       sum shape: {t.dot(t.flatten(weights), t.flatten(t.abs(target_next - target))).shape}")
        
        return -1 * ((t.dot(t.flatten(v_i_caps), t.flatten(target_next - target) * t.flatten(signs)) + t.sum(hold_cap)))


class  stock_loss_max_norm(nn.Module):
    def __init__(self, extra_node=False):
        super(stock_loss_max_norm, self).__init__()
        self.extra_node = extra_node

    def forward(self, nn_output, target, **kwargs):

        signs = nn_output / t.abs(nn_output)

        s = t.sum(t.abs(nn_output), (1, 2))

        v_i_caps = t.zeros_like(nn_output)
        for b in range(v_i_caps.shape[0]):
            v_i_caps[b,:,:] = t.abs(nn_output[b,:,:]) / s[b]

        assert t.abs(t.sum(v_i_caps) - v_i_caps.shape[0]) < 0.001

        assert t.all(v_i_caps >= 0)

        if "target_next" not in kwargs:
            target_next = t.zeros_like(target)
        else:
            target_next = kwargs["target_next"]

        target = t.abs(target)
        target_next = t.abs(target_next)

        dim = v_i_caps.dim()

        if self.extra_node:
            if dim == 2:
                hold_cap = v_i_caps[:,-1]
                v_i_caps = v_i_caps[:,:-1]
                signs = signs[:,:-1]
                # target = target[:,:-1]
                # target_next = target_next[:,:-1]
            else:
                hold_cap = v_i_caps[:,:,-1]
                v_i_caps = v_i_caps[:,:,:-1]
                signs = signs[:,:,:-1]
                # target = target[:,:,:-1]
                # target_next = target_next[:,:,:-1]
        else:
            hold_cap = t.zeros_like(v_i_caps)
        hold_cap = hold_cap.unsqueeze(1)

        # print(f"[INFO   ]       target shape: {target.shape}")
        # print(f"[INFO   ]       target next shape: {target_next.shape}")
        # print(f"[INFO   ]       weight shape: {weights.shape}")
        # print(f"[INFO   ]       hold cap shape: {hold_cap.shape}")
        # print(f"[INFO   ]       sign cap shape: {signs.shape}")
        # print(f"[INFO   ]       sum shape: {t.dot(t.flatten(weights), t.flatten(t.abs(target_next - target))).shape}")

        max_diff = t.max(t.abs(target_next - target), 2)[0]

        a = t.flatten(target_next - target) * t.flatten(signs)
        a = t.reshape(a, target.shape)
        # print(target.shape)
        b, seq, f = target.shape
        for i in range(b):
            a[i,:,:] = t.div(a[i,:,:], max_diff[i])

        # return 1.0 - (t.dot(t.flatten(v_i_caps), t.flatten(target_next - target) * t.flatten(signs) / max_diff) + t.sum(hold_cap))
        return 1.0 - (t.dot(t.flatten(v_i_caps), t.flatten(a)) + t.sum(hold_cap))
    
class  stock_loss_soft_max_norm(nn.Module):
    def __init__(self, extra_node=False):
        super(stock_loss_soft_max_norm, self).__init__()
        self.extra_node = extra_node

    def forward(self, nn_output, target, **kwargs):

        if "gamma" not in kwargs:
            gamma = 5
        else:
            gamma = kwargs["gamma"]

        signs = t.tanh(gamma * nn_output)

        s = t.sum(t.abs(nn_output), (1, 2))
        # print(s)

        v_i_caps = t.zeros_like(nn_output)
        for b in range(v_i_caps.shape[0]):
            v_i_caps[b,:,:] = t.abs(nn_output[b,:,:]) / s[b]
        # print(f"v_i_caps: {v_i_caps}")

        assert t.all(v_i_caps >= 0)
        assert t.abs(t.sum(v_i_caps) - v_i_caps.shape[0]) < 0.001

        if "target_next" not in kwargs:
            target_next = t.zeros_like(target)
        else:
            target_next = kwargs["target_next"]

        target = t.abs(target)
        target_next = t.abs(target_next)

        dim = v_i_caps.dim()

        if self.extra_node:
            if dim == 2:
                hold_cap = v_i_caps[:,-1]
                v_i_caps = v_i_caps[:,:-1]
                signs = signs[:,:-1]
                # target = target[:,:-1]
                # target_next = target_next[:,:-1]
            else:
                hold_cap = v_i_caps[:,:,-1]
                v_i_caps = v_i_caps[:,:,:-1]
                signs = signs[:,:,:-1]
                # target = target[:,:,:-1]
                # target_next = target_next[:,:,:-1]
        else:
            hold_cap = t.zeros_like(v_i_caps)
        hold_cap = hold_cap.unsqueeze(1)

        # print(f"[INFO   ]       target shape: {target.shape}")
        # print(f"[INFO   ]       target next shape: {target_next.shape}")
        # print(f"[INFO   ]       weight shape: {weights.shape}")
        # print(f"[INFO   ]       hold cap shape: {hold_cap.shape}")
        # print(f"[INFO   ]       sign cap shape: {signs.shape}")
        # print(f"[INFO   ]       sum shape: {t.dot(t.flatten(weights), t.flatten(t.abs(target_next - target))).shape}")

        max_diff = t.max(t.abs(target_next - target), 2)[0]

        a = t.flatten(target_next - target) * t.flatten(signs)
        a = t.reshape(a, target.shape)
        # print(target.shape)
        b, seq, f = target.shape
        for i in range(b):
            a[i,:,:] = t.div(a[i,:,:], max_diff[i])

        # return 1.0 - (t.dot(t.flatten(v_i_caps), t.flatten(target_next - target) * t.flatten(signs) / max_diff) + t.sum(hold_cap))
        return 1.0 - (t.dot(t.flatten(v_i_caps), t.flatten(a)) + t.sum(hold_cap))

class  stock_loss_l2_norm(nn.Module):
    def __init__(self, extra_node=False):
        super(stock_loss_l2_norm, self).__init__()
        self.extra_node = extra_node

    def forward(self, nn_output, target, **kwargs):
 
        signs = nn_output / t.abs(nn_output)

        s = t.sum(t.abs(nn_output), (1, 2))

        v_i_caps = t.zeros_like(nn_output)
        for b in range(v_i_caps.shape[0]):
            v_i_caps[b,:,:] = t.abs(nn_output[b,:,:]) / s[b]

        # print(f"V_i: {v_i_caps}")

        assert t.abs(t.sum(v_i_caps) - v_i_caps.shape[0]) < 0.001
        assert t.all(v_i_caps >= 0)

        if "target_next" not in kwargs:
            target_next = t.zeros_like(target)
        else:
            target_next = kwargs["target_next"]

        target = t.abs(target)
        target_next = t.abs(target_next)

        dim = v_i_caps.dim()

        if self.extra_node:
            if dim == 2:
                hold_cap = v_i_caps[:,-1]
                v_i_caps = v_i_caps[:,:-1]
                signs = signs[:,:-1]
                # target = target[:,:-1]
                # target_next = target_next[:,:-1]
            else:
                hold_cap = v_i_caps[:,:,-1]
                v_i_caps = v_i_caps[:,:,:-1]
                signs = signs[:,:,:-1]
                # target = target[:,:,:-1]
                # target_next = target_next[:,:,:-1]
        else:
            hold_cap = t.zeros_like(v_i_caps)
        hold_cap = hold_cap.unsqueeze(1)

        # print(f"[INFO   ]       target shape: {target.shape}")
        # print(f"[INFO   ]       target next shape: {target_next.shape}")
        # print(f"[INFO   ]       weight shape: {weights.shape}")
        # print(f"[INFO   ]       hold cap shape: {hold_cap.shape}")
        # print(f"[INFO   ]       sign cap shape: {signs.shape}")
        # print(f"[INFO   ]       sum shape: {t.dot(t.flatten(weights), t.flatten(t.abs(target_next - target))).shape}")

        max_diff = t.max(t.abs(target_next - target), 2)[0]
        # print(f"max_diff: {max_diff}")

        a = t.flatten(target_next - target) * t.flatten(signs)
        a = t.reshape(a, target.shape)
        b, seq, f = target.shape
        for i in range(b):
            a[i,:,:] = t.div(a[i,:,:], max_diff[i])
        
        # print(f"a: {a}")

        # return 1.0 - t.sqrt(t.sum(
        #     t.dot(t.flatten(v_i_caps), t.pow(t.flatten(target_next - target) / max_diff, 2))
        #     + t.flatten(hold_cap)*t.flatten(hold_cap)
        # ))
        return 1.0 - t.sqrt(
            t.dot(t.flatten(v_i_caps), t.pow(t.flatten(a), 2))
            + t.sum(t.flatten(hold_cap)*t.flatten(hold_cap))
        )

class  stock_loss_global_norm(nn.Module):
    def __init__(self, extra_node=False):
        super(stock_loss_global_norm, self).__init__()
        self.extra_node = extra_node

    def forward(self, nn_output, target, **kwargs):

        signs = nn_output / t.abs(nn_output)

        s = t.sum(t.abs(nn_output), (1, 2))

        v_i_caps = t.zeros_like(nn_output)
        for b in range(v_i_caps.shape[0]):
            v_i_caps[b,:,:] = t.abs(nn_output[b,:,:]) / s[b]

        assert t.abs(t.sum(v_i_caps) - v_i_caps.shape[0]) < 0.001

        assert t.all(v_i_caps >= 0)

        if "target_next" not in kwargs:
            target_next = t.zeros_like(target)
        else:
            target_next = kwargs["target_next"]

        target = t.abs(target)
        target_next = t.abs(target_next)

        # max_diff = t.max(t.abs(target_next - target))

        dim = nn_output.dim()

        if self.extra_node:
            if dim == 2:
                nn_output = nn_output[:,:-1]
                signs = signs[:,:-1]
                # target = target[:,:-1]
                # target_next = target_next[:,:-1]
            else:
                nn_output = nn_output[:,:,:-1]
                signs = signs[:,:,:-1]
                # target = target[:,:,:-1]
                # target_next = target_next[:,:,:-1]
        # else:
        #     hold_cap = t.zeros_like(nn_output)
        # hold_cap = hold_cap.unsqueeze(1)

        # print(f"[INFO   ]       target shape: {target.shape}")
        # print(f"[INFO   ]       target next shape: {target_next.shape}")
        # print(f"[INFO   ]       weight shape: {weights.shape}")
        # print(f"[INFO   ]       hold cap shape: {hold_cap.shape}")
        # print(f"[INFO   ]       sign cap shape: {signs.shape}")
        # print(f"[INFO   ]       sum shape: {t.dot(t.flatten(weights), t.flatten(t.abs(target_next - target))).shape}")

        # print(f"nn_output: {nn_output}")
        # print(f"signs: {signs}")
        # print(f"target_next: {target_next}")
        # print(f"target: {target}")
        # print(f"target_next - target: {target_next - target}")
        
        denom = t.dot(t.flatten(t.abs(nn_output)), t.flatten(target_next - target))

        # print(f"Denom: {denom}")
        # print(f"product: {t.flatten(t.abs(nn_output)) * t.flatten(target_next - target) * t.flatten(signs)}")
        # print(f"norm product: {- t.sum(t.flatten(t.abs(nn_output)) * t.flatten(target_next - target) * t.flatten(signs) / denom)}")

        return - t.sum(
            t.flatten(t.abs(nn_output)) * t.flatten(target_next - target) * t.flatten(signs) / denom
            )