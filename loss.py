import torch
import torch.nn as nn
import torch.nn.functional as F

# write a cross-entropy loss function for the classifier


def cross_entropy_loss(result, gt):
    """
    Compute the cross entropy loss for the classifier

    :param
    result: the output of the classifier
    gt: the ground truth labels
    """
    return F.cross_entropy(result, gt)

def accuracy(result, gt):
    """
    Compute the accuracy for the classifier

    :param
    result: the output of the classifier
    gt: the ground truth labels
    """
    return (result.argmax(dim=1) == gt).float().mean()