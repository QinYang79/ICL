from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn 
import torch.nn.functional as F

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size, text_length=args.text_length)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
 
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
 
        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
 
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()
 
    def encode_text_tse(self, text):
        x, atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

 
    def encode_image_full(self, image):
        x,atten_i = self.base_model.encode_image(image)
        a = x[:, 0, :].float()
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return a, i_tse_f.float()
    
    def encode_text_full(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        a = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        return a, t_tse_f.float()

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB

    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        rcaption_ids = batch['rcaption_ids']
        
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            
        label_hat = batch['label_hat'].to(i_feats.device) 
        if 'aug' in self.current_task:

            x, atten_t_r = self.base_model.encode_text(rcaption_ids.long())
            t_feats_r = x[torch.arange(rcaption_ids.shape[0]), rcaption_ids.argmax(dim=-1)].float()
            t_tse_f_r = self.texual_emb_layer(x, rcaption_ids, atten_t_r) # keep nan

            i_feats = i_feats / i_feats.norm(dim=-1, keepdim=True)
            t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
            i_tse_f = i_tse_f / i_tse_f.norm(dim=-1, keepdim=True)
            t_tse_f = t_tse_f / t_tse_f.norm(dim=-1, keepdim=True)
            t_feats_r = t_feats_r / t_feats_r.norm(dim=-1, keepdim=True)
            t_tse_f_r = t_tse_f_r / t_tse_f_r.norm(dim=-1, keepdim=True)
            
            sims1_base = i_feats @ t_feats.t()
            sims2_base = i_tse_f @ t_tse_f.t()
            
            sims1_aug = i_feats @ t_feats_r.t()
            sims2_aug = i_tse_f @ t_tse_f_r.t() 
            
            labels1 = labels2 = label_hat 
            lllam = max([0.1, 1 - 0.03 * self.epoch]) 
 
            loss1 = (objectives.compute_TAL_per(sims1_base, batch['pids'], self.args.tau, margin=self.args.margin) * label_hat).sum()
            loss2 = (objectives.compute_TAL_per(sims2_base, batch['pids'], self.args.tau, margin=self.args.margin) * label_hat).sum()
            ret.update({'bge_loss':loss1})
            ret.update({'tse_loss':loss2})

            loss11 = (objectives.compute_TAL_per(sims1_aug, batch['pids'], self.args.tau, margin=self.args.margin) * labels1).sum() * lllam  
            loss22 = (objectives.compute_TAL_per(sims2_aug, batch['pids'], self.args.tau, margin=self.args.margin) * labels2).sum() * lllam
            ret.update({'bge_aug_loss':loss11})
            ret.update({'tse_aug_loss':loss22}) 
        else:
            loss1, loss2, sims_diag = objectives.compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'], \
            label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
            loss_type=self.loss_type,logit_scale=self.logit_scale)

            ret.update({'bge_loss':loss1})
            ret.update({'tse_loss':loss2})

        return ret

class CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._set_task()
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size, text_length=args.text_length)
  
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x,_ = self.base_model.encode_image(image)
        return x[:, 0, :].float() 

    def encode_text(self, text):
        x,_ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model

def build_clip_model(args):
    model = CLIP(args)
    # covert model to fp16
    convert_weights(model)
    return model
