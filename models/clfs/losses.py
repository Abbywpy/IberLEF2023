
from loguru import logger
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_printoptions(profile="full")


# write a cross-entropy loss function for the classifier


def accuracy(result, gt):
    """
    Compute the accuracy for the classifier

    :param
    result: the output of the classifier
    gt: the ground truth labels
    """
    return (result.argmax(dim=1) == gt).float().mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
 
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()


        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class LossFunctions():
    def __init__(self, eps=1e-8):
        self.eps = eps

    def mean_squared_error(self, real, predictions):
        loss = (real - predictions).pow(2)
        return loss.sum(-1).mean()

    def reconstruction_loss(self, real, predicted, rec_type='mse'):
        if rec_type == 'mse':
            loss = (real - predicted).pow(2)
        elif rec_type == 'bce':
            loss = F.binary_cross_entropy(predicted, real, reduction='none')
        else:
            raise "invalid loss function... try bce or mse..."
        return loss.sum(-1).mean()

    def log_normal(self, x, mu, var):
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)

    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        loss = self.log_normal(z, z_mu, z_var) - \
            self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()

    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))


def build_multi_classification_loss(predictions, labels):
    shape = tuple(labels.shape)
    labels = labels.float()
    y_i = torch.eq(labels, torch.ones(shape))
    y_not_i = torch.eq(labels, torch.zeros(shape))

    truth_matrix = pairwise_and(y_i, y_not_i).float()
    sub_matrix = pairwise_sub(predictions, predictions)
    exp_matrix = torch.exp(-5*sub_matrix)
    sparse_matrix = exp_matrix * truth_matrix
    sums = torch.sum(sparse_matrix, dim=[2, 3])
    y_i_sizes = torch.sum(y_i.float(), dim=1)
    y_i_bar_sizes = torch.sum(y_not_i.float(), dim=1)
    normalizers = y_i_sizes * y_i_bar_sizes
    loss = torch.div(sums, 5*normalizers)  # 100*128  divide  128
    zero = torch.zeros_like(loss)  # 100*128 zeros
    loss = torch.where(torch.logical_or(
        torch.isinf(loss), torch.isnan(loss)), zero, loss)
    loss = torch.mean(loss)
    return loss


def pairwise_and(a, b):
    column = torch.unsqueeze(a, 2)
    row = torch.unsqueeze(b, 1)
    return torch.logical_and(column, row)


def pairwise_sub(a, b):
    column = torch.unsqueeze(a, 3)
    row = torch.unsqueeze(b, 2)
    return column - row


def cross_entropy_loss(logits, labels, n_sample):
    labels = torch.tile(torch.unsqueeze(labels, 0), [n_sample, 1, 1])
    ce_loss = nn.BCEWithLogitsLoss(labels=labels, logits=logits)
    ce_loss = torch.mean(torch.sum(ce_loss, dim=1))
    return ce_loss


def compute_loss(input_label, output, args=None):
    if args.reg == "gumbel":
        fe_out, fe_mu, fe_logvar, label_emb = output['label_out'], output[
            'fe_mu'], output['fe_logvar'], output['label_emb']
        fx_out, fx_mu, fx_logvar, feat_emb = output['feat_out'], output[
            'fx_mu'], output['fx_logvar'], output['feat_emb']
        fx_out2, single_label_out = output['feat_out2'], output['single_label_out']
        embs = output['embs']
        feat_recon, feat = output['feat_recon'], output['feat']
        losses = LossFunctions()
        fx_loss_cat = - \
            losses.entropy(output['fx_gumbel_logits'],
                           output['fx_gumbel_prob']) - np.log(0.1)
        fe_loss_cat = - \
            losses.entropy(output['fe_gumbel_logits'],
                           output['fe_gumbel_prob']) - np.log(0.1)
    else:
        fe_out, fe_mu, fe_logvar, label_emb = output['label_out'], output[
            'fe_mu'], output['fe_logvar'], output['label_emb']
        fx_out, fx_mu, fx_logvar, feat_emb = output['feat_out'], output[
            'fx_mu'], output['fx_logvar'], output['feat_emb']
        fx_out2, single_label_out = output['feat_out2'], output['single_label_out']
        embs = output['embs']

    feat_recon_loss = 0.

    latent_cpc_loss = 0.
    if args.reg == "gmvae":
        fe_sample = torch.matmul(input_label, fe_mu) / \
            input_label.sum(1, keepdim=True)
        latent_cpc_loss = SupConLoss(temperature=0.1)(
            torch.stack([fe_sample, fx_mu], dim=1), input_label.float())
    else:
        latent_cpc_loss = SupConLoss(temperature=0.1)(
            torch.stack([fe_mu, fx_mu], dim=1), input_label.float())

    if args.reg == "vae":
        kl_loss = torch.mean(0.5*torch.sum((fx_logvar-fe_logvar)-1+torch.exp(
            fe_logvar-fx_logvar)+torch.square(fx_mu-fe_mu)/(torch.exp(fx_logvar)+1e-6), dim=1))
    elif args.reg == "btcae":
        std = torch.exp(0.5*fx_logvar)
        eps = torch.randn_like(std)
        fx_sample = fx_mu + eps*std
        kl_loss = KL((fe_mu, fe_logvar), fx_sample)
    elif args.reg == "wae":
        kl_loss = 0.
        for i in range(10):
            std = torch.exp(0.5*fx_logvar)
            eps = torch.randn_like(std)
            fx_sample = fx_mu + eps*std
            kl_loss += imq_kernel(fe_mu, fx_sample, h_dim=fx_mu.shape[1])
        kl_loss /= 10.
        kl_loss *= 5
    elif args.reg == "gmvae":
        std = torch.exp(0.5*fx_logvar)
        eps = torch.randn_like(std)
        fx_sample = fx_mu + eps*std
        fx_var = torch.exp(fx_logvar)
        fe_var = torch.exp(fe_logvar)
        kl_loss = (log_normal(fx_sample, fx_mu, fx_var) -
                   log_normal_mixture(fx_sample, fe_mu, fe_var, input_label)).mean()

    def compute_BCE_and_RL_loss(E):
        # compute negative log likelihood (BCE loss) for each sample point
        sample_nll = -(torch.log(E)*input_label+torch.log(1-E)*(1-input_label))

        logprob = -torch.sum(sample_nll, dim=2)

        # the following computation is designed to avoid the float overflow (log_sum_exp trick)
        maxlogprob = torch.max(logprob, dim=0)[0]
        Eprob = torch.mean(torch.exp(logprob-maxlogprob), axis=0)
        nll_loss = torch.mean(-torch.log(Eprob)-maxlogprob)

        #c_loss = build_multi_classification_loss(E, input_label)
        return nll_loss

    def supconloss(label_emb, feat_emb, embs, temp=1.0, sample_wise=False):
        if sample_wise:
            loss_func = SupConLoss(temperature=0.1)
            return loss_func(torch.stack([label_emb, feat_emb], dim=1), input_label.float())

        features = torch.cat((label_emb, feat_emb))
        labels = torch.cat((input_label, input_label)).float()
        n_label = labels.shape[1]
        emb_labels = torch.eye(n_label).to(feat_emb.device)
        mask = torch.matmul(labels, emb_labels)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, embs),
            temp)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        mean_log_prob_neg = ((1.0-mask) * log_prob).sum(1) / (1.0-mask).sum(1)

        loss = - mean_log_prob_pos
        loss = loss.mean()

        return loss

    if not args.finetune:
        pred_e = torch.sigmoid(fe_out)
        pred_x = torch.sigmoid(fx_out)
        pred_x2 = torch.sigmoid(fx_out2)
        pred_single_label = torch.sigmoid(single_label_out)
        single_label_recon_loss = nn.BCELoss()(pred_single_label, torch.eye(
            pred_single_label.shape[1]).to(pred_single_label.device))

        nll_loss = compute_BCE_and_RL_loss(pred_e.unsqueeze(0))
        nll_loss_x = compute_BCE_and_RL_loss(pred_x.unsqueeze(0))
        nll_loss_x2 = compute_BCE_and_RL_loss(pred_x2.unsqueeze(0))
        cpc_loss = supconloss(label_emb, feat_emb, embs,
                              sample_wise=args.cpc_sample_wise)
        total_loss = (nll_loss + nll_loss_x + nll_loss_x2) * \
            args.nll_coeff + kl_loss*6. + (cpc_loss)  # + latent_cpc_loss
    else:
        pred_x = torch.sigmoid(fx_out)
        pred_e = torch.ones_like(pred_x)
        nll_loss_x, c_loss_x = compute_BCE_and_RL_loss(pred_x.unsqueeze(0))
        nll_loss, c_loss = 0., 0.
        total_loss = nll_loss_x * args.nll_coeff + c_loss_x * args.c_coeff + kl_loss

    return total_loss, nll_loss, nll_loss_x, 0., 0., kl_loss, cpc_loss, latent_cpc_loss, single_label_recon_loss, pred_e, pred_x
