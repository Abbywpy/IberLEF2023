import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from models.clfs.losses import compute_loss

from loguru import logger


class mlclCLF(nn.Module):
    """
    Multi-label Contrastive Learning (MLCL) module.
    - contrastive training (learning to project embeddings in a semantically-meaningful space)
    """

    def __init__(self, latent_dim, **kargs):
        super(mlclCLF, self).__init__()
        logger.info(latent_dim)
        self.vae = VAE(latent_dim = latent_dim)

    def forward(self, concated_embeds, input_label, **args):
        output = self.vae(input_label, concated_embeds)

        total_loss, nll_loss, nll_loss_x, c_loss, c_loss_x, kl_loss, cpc_loss, latent_cpc_loss, feat_recon_loss, _, pred_x = compute_loss(
            input_label, output, args)
        return {"pred_output": output, "total_loss": total_loss}


class VAE(nn.Module):
    # batch_size, latent_dim = latent_sample.shape
    def __init__(self, latent_dim, feature_dim=1000, meta_offset=0,  emb_size=2048, label_dim=38, keep_prob=0.5, scale_coeff=1.0, **kargs):
        super(VAE, self).__init__()
        # feature layers
        input_dim = feature_dim + meta_offset
        self.fx1 = nn.Linear(input_dim, 256)
        self.fx2 = nn.Linear(256, 512)
        self.fx3 = nn.Linear(512, 256)
        self.fx_mu = nn.Linear(256, latent_dim)
        self.fx_logvar = nn.Linear(256, latent_dim)

        self.emb_size = emb_size

        self.fd_x1 = nn.Linear(input_dim+latent_dim, 512)
        self.fd_x2 = torch.nn.Sequential(
            nn.Linear(512, self.emb_size)
        )
        self.feat_mp_mu = nn.Linear(self.emb_size, label_dim)

        self.recon = torch.nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

        self.label_recon = torch.nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.emb_size),
            nn.LeakyReLU()
        )

        # label layers
        self.fe0 = nn.Linear(label_dim, self.emb_size)
        self.fe1 = nn.Linear(self.emb_size, 512)
        self.fe2 = nn.Linear(512, 256)
        self.fe_mu = nn.Linear(256, latent_dim)
        self.fe_logvar = nn.Linear(256, latent_dim)

        self.fd1 = self.fd_x1
        self.fd2 = self.fd_x2
        self.label_mp_mu = self.feat_mp_mu

        self.bias = nn.Parameter(torch.zeros(label_dim))

        assert id(self.fd_x1) == id(self.fd1)
        assert id(self.fd_x2) == id(self.fd2)

        # things they share
        self.dropout = nn.Dropout(p=keep_prob)
        self.scale_coeff = scale_coeff

    def label_encode(self, x):
        h0 = self.dropout(F.relu(self.fe0(x)))
        h1 = self.dropout(F.relu(self.fe1(h0)))
        h2 = self.dropout(F.relu(self.fe2(h1)))
        mu = self.fe_mu(h2) * self.scale_coeff
        logvar = self.fe_logvar(h2) * self.scale_coeff
        fe_output = {
            'fe_mu': mu,
            'fe_logvar': logvar
        }
        return fe_output

    def feat_encode(self, x):
        h1 = self.dropout(F.relu(self.fx1(x)))
        h2 = self.dropout(F.relu(self.fx2(h1)))
        h3 = self.dropout(F.relu(self.fx3(h2)))
        mu = self.fx_mu(h3) * self.scale_coeff
        logvar = self.fx_logvar(h3) * self.scale_coeff
        fx_output = {
            'fx_mu': mu,
            'fx_logvar': logvar
        }

        return fx_output

    def label_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def feat_reparameterize(self, mu, logvar, coeff=1.0):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def label_decode(self, z):
        d1 = F.relu(self.fd1(z))
        d2 = F.leaky_relu(self.fd2(d1))
        d3 = F.normalize(d2, dim=1)
        return d3

    def feat_decode(self, z):
        d1 = F.relu(self.fd_x1(z))
        d2 = F.leaky_relu(self.fd_x2(d1))
        d3 = F.normalize(d2, dim=1)
        return d3

    def label_forward(self, x, feat):
        n_label = x.shape[1]
        all_labels = torch.eye(n_label).to(x.device)
        fe_output = self.label_encode(all_labels)
        mu = fe_output['fe_mu']
        logvar = fe_output['fe_logvar']

        z = torch.matmul(x, mu) / x.sum(1, keepdim=True)

        label_emb = self.label_decode(torch.cat((feat, z), 1))
        single_label_emb = F.normalize(self.label_recon(mu), dim=1)

        fe_output['label_emb'] = label_emb
        fe_output['single_label_emb'] = single_label_emb
        return fe_output

    def feat_forward(self, x):
        fx_output = self.feat_encode(x)
        mu = fx_output['fx_mu']
        logvar = fx_output['fx_logvar']

        z = self.feat_reparameterize(mu, logvar)
        z2 = self.feat_reparameterize(mu, logvar)

        feat_emb = self.feat_decode(torch.cat((x, z), 1))
        feat_emb2 = self.feat_decode(torch.cat((x, z2), 1))
        fx_output['feat_emb'] = feat_emb
        fx_output['feat_emb2'] = feat_emb2

        feat_recon = self.recon(z)
        fx_output['feat_recon'] = feat_recon
        return fx_output

    def forward(self, label, feature):
        fe_output = self.label_forward(label, feature)
        label_emb, single_label_emb = fe_output['label_emb'], fe_output['single_label_emb']
        fx_output = self.feat_forward(feature)
        feat_emb, feat_emb2 = fx_output['feat_emb'], fx_output['feat_emb2']
        embs = self.fe0.weight

        label_out = torch.matmul(label_emb, embs)
        single_label_out = torch.matmul(single_label_emb, embs)
        feat_out = torch.matmul(feat_emb, embs)
        feat_out2 = torch.matmul(feat_emb2, embs)

        fe_output.update(fx_output)
        output = fe_output
        output['embs'] = embs
        output['label_out'] = label_out
        output['single_label_out'] = single_label_out
        output['feat_out'] = feat_out
        output['feat_out2'] = feat_out2
        output['feat'] = feature

        return output
