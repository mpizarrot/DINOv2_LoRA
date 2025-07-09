import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .dinov2.layers.lora import LoRALayer
from src.dinov2.models.vision_transformer import vit_base
from experiments.options import opts

from .utils import mark_only_lora_as_trainable, apply_lora


def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

def lora_trainable(model, bias='all'):
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError
            

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts

        self.dino = vit_base(patch_size=14, block_chunks=0, init_values=1.0) 
        apply_lora(self.opts, self.dino)
        print("LoRA applied")
        print(self.dino)
        self.dino.apply(lora_trainable)

        # Prompt Learning
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.best_metric = 1e3

    def configure_optimizers(self):
        model_params = list(self.dino.parameters())

        if self.opts.prompt_learning is not False:
            model_params += [self.sk_prompt, self.img_prompt]

        optimizer = torch.optim.Adam(model_params, lr=self.opts.encoder_lr)
        return optimizer
    
    def loss_fn_nx(self, z1, z2, temperature=0.5):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N, Z = z1.shape
        device = z1.device
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
        
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
        diag = torch.eye(2*N, dtype=torch.bool, device=device)
        diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

        negatives = similarity_matrix[~diag].view(2*N, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= temperature

        labels = torch.zeros(2*N, device=device, dtype=torch.int64)

        loss = F.cross_entropy(logits, labels, reduction='sum')
        return loss / (2 * N)
    
    def loss_clip(self, emb_sketch, emb_photo):
        norm_emb_sketch = F.normalize(emb_sketch, dim=1)
        norm_emb_photo = F.normalize(emb_photo, dim=1)

        similarity_matrix = norm_emb_sketch @ norm_emb_photo.T
        loss = F.cross_entropy(similarity_matrix, torch.arange(similarity_matrix.shape[0], device=self.device), reduction='none')
        return loss.mean()
    
    def loss_fn_nc(self, emb_sketch, emb_photo):
        sketch_soft = F.softmax(emb_sketch, dim=1)
        photo_soft = F.softmax(emb_photo, dim=1)

        loss = -torch.sum(photo_soft * torch.log(sketch_soft))

        return loss
    
    def info_nce_loss(self, emb_sketch, emb_image, temperature=0.5):
        img_norm = F.normalize(emb_image)
        sketch_norm = F.normalize(emb_sketch)
        
        sim = img_norm @ sketch_norm.T
        mask = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        
        sim.masked_fill_(mask, -9e15) # los positivos == -9e15
        
        pos_mask = mask.roll(shifts=sim.shape[0] // 2, dims=0)

        # InfoNCE loss
        sim = sim / temperature
        nll = -sim[pos_mask] + torch.logsumexp(sim, dim=-1)
        nll = nll.mean()
        return nll
    
    def loss_emb_cos(self, emb_sketch, emb_photo):
        return self.emb_cos_loss(emb_sketch, emb_photo, torch.ones(emb_sketch.shape[0], device=self.device))
    
    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def loss_barlowtwins(self, y1, y2):
        z1 = (y1 - y1.mean(0)) / y1.std(0)  # F.normalize(y1, dim=0) #self.projector(self.backbone(y1))
        z2 = (y2 - y2.mean(0)) / y2.std(0)   # F.normalize(y2, dim=0)  #self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = (z1.T @ z2) / y1.shape[0]

        # c_diff = (c - torch.eye(c.shape[1], device=c.device)).pow(2) # dimXdim
        # off_diag = self.off_diagonal(c_diff).mul_(0.0051)
        # loss = off_diag.sum()

        # sum the cross-correlation matrix between all gpus
        # c.div_(self.args.batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + 0.0051 * off_diag
        return loss
        
    
    def loss_fn(self, sketch_emb, image_emb):
        n_loss_terms = 0
        total_loss = 0
        sketch_emb_out = F.softmax(sketch_emb / 0.5, dim=1)

        for iq, q in enumerate(image_emb):
            for v in range(len(sketch_emb_out)):
                if iq == v:
                    continue
                loss = torch.sum(-q * F.log_softmax(sketch_emb_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        return total_loss / n_loss_terms

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.dino(data, prompt=self.img_prompt.expand(
                data.shape[0], -1, -1) if self.opts.prompt_learning is not False else None)
        else:
            feat = self.dino(data, prompt=self.sk_prompt.expand(
                data.shape[0], -1, -1) if self.opts.prompt_learning is not False else None)
        return feat

    def training_step(self, batch, batch_idx):
        img_anchor, img_positive, img_negative = batch[:3]
        img_anchor_feat = self.forward(img_anchor, dtype='image')
        img_positive_feat = self.forward(img_positive, dtype='image')
        img_negative_feat = self.forward(img_negative, dtype='image')

        loss = self.loss_fn_triplet(img_anchor_feat, img_positive_feat, img_negative_feat)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img_anchor, img_positive, img_negative = batch[:3]
        img_anchor_feat = self.forward(img_anchor, dtype='image')
        img_positive_feat = self.forward(img_positive, dtype='image')
        img_negative_feat = self.forward(img_negative, dtype='image')

        loss = self.loss_fn_triplet(img_anchor_feat, img_positive_feat, img_negative_feat)
        self.log('val_loss', loss)
        return img_anchor_feat, img_positive_feat, None # img_negative_feat

    def validation_epoch_end(self, val_step_outputs):
        Len = len(val_step_outputs)
        if Len == 0:
            return
        
        anchor_feat_all = torch.cat([val_step_outputs[i][0] for i in range(Len)])
        positive_feat_all = torch.cat([val_step_outputs[i][1] for i in range(Len)])

        # Aplicamos la perdida de clip
        loss_clip = self.loss_clip(anchor_feat_all, positive_feat_all)

        self.log('clip_loss', loss_clip)
        if self.global_step > 0:
            self.best_metric = self.best_metric if  (self.best_metric < loss_clip.item()) else loss_clip.item()
        print ('loss_clip: {}, Best loss_clip: {}'.format(loss_clip.item(), self.best_metric))