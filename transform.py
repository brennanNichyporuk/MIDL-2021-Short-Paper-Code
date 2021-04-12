from scipy import ndimage
import numpy as np
import torch


# ReWeighing Transform
cca_structure = ndimage.generate_binary_structure(3, 2)
class ReWeighingTransform:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta  = beta

    def __call__(self, data_dict):
        target = data_dict['TARGET']
        weight = np.ones(target.shape)
        
        cca, n = ndimage.label(target, structure=cca_structure)
        for i in range(1, n+1):
            lesion_i = cca==i
            size = np.sum(lesion_i)
            weight[lesion_i] = 1 + (self.alpha / size) * (np.e **
                    ((-1/self.beta) * (size-1)))
        data_dict['WEIGHT'] = weight
        return data_dict


# Define Criterion
bce = nn.BCEWithLogitsLoss(reduction='none')
def reweighting_criterion(output, target, mask, weight):
    is_mask = mask==1
    
    loss = []
    for (_output, _target, _is_mask, _weight) in zip(output, target, is_mask, weight):
        loss.append(torch.mean(_weight[_is_mask] * bce(_output[_is_mask],
            _target[_is_mask])))
    loss = torch.mean(torch.stack(loss))
    return loss
