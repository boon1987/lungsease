

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import copy


def calculate_multilabel_binary_class_weight(targets):
    # Compute balanced class weight for multilabel classification

    class_weights=[]
    for i in np.arange(targets.shape[1]):
        temp_y = targets[:,i]
        class_weights.append(list(compute_class_weight('balanced', np.array([0,1]), y=temp_y)))
    class_weights = torch.as_tensor(copy.deepcopy(np.array(class_weights).transpose()))
    return class_weights


class Custom_MultiLabel_AlphaBalanced_FocalLoss(torch.nn.Module): 
    """
    Focal Loss: Modified based on the libauc focal loss
    Reference: https://amaarora.github.io/2020/06/29/FocalLoss.html
    """
    def __init__(self, alpha=None, gamma=2, num_classes=5):
        super(Custom_MultiLabel_AlphaBalanced_FocalLoss, self).__init__()

        # alpha is class weight for binary class classification task
        if alpha is None:
            self.alpla = (torch.tensor([1.0]*5)).repeat(2, 1)
            self.alpha = self.alpla.cuda()
        else:
            self.alpha = torch.tensor(alpha).cuda()

        # gamma is parameter for smoothing the hardness of sample. 
        self.gamma = torch.tensor(gamma).cuda()

    def forward(self, inputs, targets):

            
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)

        # Modified for multilabel classification task compared to standard pytorch binary class focal loss
        targets = targets.type(torch.long)
        at = self.alpha.gather(dim=0, index=targets)
        F_loss = at.view(-1)*(1-pt.view(-1))**self.gamma * BCE_loss.view(-1)

        # print(self.alpha)
        # print(targets)
        # print('at',at)
        # print('pt',pt)
        # print('BCE_loss', BCE_loss)

        return F_loss.mean()