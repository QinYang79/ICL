import argparse
import base64
from io import BytesIO
import json
import random
import re
import time
from datasets import build_dataloader 
from datasets.bases import TextPureDataset  
from prettytable import PrettyTable
import copy
import torch
import numpy as np 
import torch.nn.functional as F
import logging 
from vllm import LLM, SamplingParams
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.logger import setup_logger
from utils.iotools import load_train_configs
from model import build_model, build_clip_model
from utils.checkpoint import Checkpointer
import os.path as op
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info 
from openai import OpenAI    
import time

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]
    
    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    if retur_indices:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]], indices
    else:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]]
 
def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class MLLMs(object):
    def __init__(self, model_dir= "/home/qinyang/projects/2024-CVPR-RDE/LLMs/Qwen2-VL-7B-Instruct"):
        
        self.model_dir = model_dir  
        print("Loading ...............")
        print(model_dir)
        self.llm = LLM(model=model_dir,
                    tensor_parallel_size = 2, 
                    gpu_memory_utilization = 0.8, 
                    # max_model_len = 1024,
                    # max_num_seqs = 128,
                    dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        print("Loaded ...............")

    def generate_response_multi_images(self, questions, images=None, sys = "You are a helpful assistant.", t=0.01):
        messages = [ 
            [{"role": "system", "content": sys}, {"role": "user",  "content": [
                {
                    "type": "image", 
                    "image": images[i], 
                    "min_pixels": 50176,
                    "max_pixels": 50176, 
                },
                {"type": "text", "text": p}
                ]}] 
            for i, p in enumerate(questions)]
        prompts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]   
        image_data = [process_vision_info(msg)[0] for msg in messages]
    
        inputs = [{
            "prompt": p,
            "multi_modal_data": { 
                'image': image_data[i]
            } 
            
        } for i, p in enumerate(prompts)]
        sampling_params = SamplingParams(temperature=t, max_tokens=2048, skip_special_tokens=True) 
        outputs = self.llm.generate(inputs, sampling_params=sampling_params) 
        return [o.outputs[0].text for o in outputs] 
    
    def generate_response_qwen2vl(self, questions, sys = "You are a helpful assistant.", t=0.01): 
        messages = [ 
            [{"role": "system", "content": sys}, {"role": "user",  "content": [{"type": "text", "text": p}]}] 
            for i, p in enumerate(questions)]
        prompts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages] 
        inputs = [{
                "prompt": p,  
        } for i, p in enumerate(prompts)]
        sampling_params = SamplingParams(temperature=t, max_tokens=2048, skip_special_tokens=True) 
        outputs = self.llm.generate(inputs, sampling_params=sampling_params) 
        return [o.outputs[0].text for o in outputs] 

  
def get_pretrained_model(args_model):
    # CUHK-PEDES ICFG-PEDES RSTPReid
    if base_model == 'RDE':
        model = build_model(args_model, num_classes)
    else:
        model = build_clip_model(args_model)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(paras['embed_model_path'], 'best.pth'))
    model = model.cuda() 
    model.eval()
    return model

class ImgDataset(Dataset):
    def __init__(self, images):  
        self.images = images
    def __len__(self):
        return len(self.images) 
    def __getitem__(self, index):
        return {
            'index':index,
            'images': load_image(self.images[index])
        }
class TxtDataset(Dataset):
    def __init__(self, captions):  
        self.captions = captions
    def __len__(self):
        return len(self.captions) 
    def __getitem__(self, index):
        return {
            'index':index,
            'captions': self.captions[index]
        }
        
def collate(batch):
    keys = set([key for b in batch for key in b.keys()]) 
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
            batch_tensor_dict.update({k: torch.stack(v)})
        else:
            batch_tensor_dict.update({k: list(v)})  

    return batch_tensor_dict

def read_json_from_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def process_cap_(caps):
    tmps = []
    for c in caps:
        c = c.split('\n') 
        tmp = []
        for cc in c:
            if ':' in cc:
                continue
            try:
                cc = cc.split('. ')[1]
                if 'Yes, ' in cc:
                    cc  = cc.replace('Yes, ','')
                if 'No, ' in cc:
                    cc  = cc.replace('No, ','')
                    
                cc = cc[:1].upper() + cc[1:]
                if cc[-1:] != '.':
                    cc += '.'
            except:
                cc = '' 
            tmp.append(cc) 
        tmps.append(tmp) 
    return tmps

def print_rs(sims_dict, qids, pids):
    table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP","rSum"])
    for key in sims_dict.keys():
        sims = sims_dict[key]
        rs = get_metrics(sims, qids, pids, f'{key}-t2i',False)
        table.add_row(rs) 

    table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
    table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
    table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
    table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
    table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
    table.custom_format["RSum"] = lambda f, v: f"{v:.2f}"
    logger.info('\n' + str(table))

def batch_infer(llm, b_prompts, images,  t=0.2):
    results = []
    n_samples = len(b_prompts)
    n_batches = (len(b_prompts) - 1)//batch_size + 1
    results = []
    for i in tqdm(range(n_batches)): 
        start = i*batch_size
        end = n_samples if i== n_batches -1 else (i+1)*batch_size   
        rs = llm.generate_response_multi_images(questions=b_prompts[start:end], images=images[start:end], t=t)
        print(rs[0])
        results += rs
    return results

def batch_infer_txt(llm, b_prompts,  t=0.2):
    results = []
    n_samples = len(b_prompts)
    n_batches = (len(b_prompts) - 1)//batch_size + 1
    results = []
    for i in tqdm(range(n_batches)): 
        start = i*batch_size
        end = n_samples if i== n_batches -1 else (i+1)*batch_size   
        rs = llm.generate_response_qwen2vl(questions=b_prompts[start:end], t=t)
        print(rs[0])
        results += rs
    return results

def round_llm(args, round, llm): 
    k = n_round
    
    global gt_indexs, rcaptions, xi, sims_base, decisions 
    results = []
    n_samples = sims_base.size(0)
    n_batches = (n_samples - 1)//batch_size + 1
    agrmaxs = []
    for i in tqdm(range(n_batches)): 
        start = i*batch_size
        end = n_samples if i== n_batches -1 else (i+1)*batch_size   
        agrmaxs += sims_base[start:end].topk(dim=1, k=k)[1].numpy().tolist() 
    
    ref_images = [[img_paths[j] for j in i] for i in agrmaxs] 
    ref_indexs = [[j for j in i] for i in agrmaxs] 
    ref_pids = [[pids[j]  for j in i] for i in agrmaxs]
 
    prompt1 = """Can this text accurately describe the image?

Text: {cap}

Answer "Yes" or "No"."""

    prompt2 = """According to the pedestrian image, answer the following questions one by one:

1. The person is male or female?
2. What hairstyle does the person have, such as hair length and color?
3. What is this person wearing on his upper body? If clearly visible, what are the color, type, and sleeve length?
4. What are the characteristics of this person's pants? If clearly visible, what are the color, type, and trouser leg length?
5. Does this person have any patterns on his/her clothes or pants?
6. What are the characteristics of this person's shoes? If clearly visible, what are the color and style?
7. Does this person wear glasses? If clearly visible, what are the color and style?
8. Is this person wearing a scarf? If clearly visible, what are the color and style?
9. Does this person have something in his/her hand? If so, what is it and what color is it?
10. Does this person carry a backpack? If clearly visible, what are the color and style?
11. Does this person wear a hat? If clearly visible, what are the color and style?
12. Is this person wearing a belt or waistband?
13. What is this person doing?
14. What is the background?
15. Are there other people in the background of this person?"""
 
    prompt3 = """Aggregate the following subtexts into continuous and concise text sentences.
    
Example:
Subtexts: ['The person is wearing a black jacket with a white stripe on the sleeve.', 'The person is male.', 'The person has short brown hair.', 'The person is wearing a sleeveless striped shirt and a green tank top.', 'The person is wearing green pants.', 'The person is wearing green pants.', 'The person is wearing a red scarf around their neck.', 'The person is wearing a red hat on their head.', 'The background is an outdoor area with some structures and other people.']
Output: The man has short brown hair and is wearing a black jacket with a white stripe on the sleeve, a sleeveless striped shirt, a green tank top, green pants, a red scarf around his neck, and a red hat. The background features an outdoor area with some structures and other people.

Now let's get started.
Subtexts: {cap}

Output aggregated sentences without any explanation."""
 
    top_indexs = [arg[0] for arg in agrmaxs] #top1 
    sims = [sims_base[i][ref_indexs[i][0]] for i, v in enumerate(gt_indexs)]
    conditions = [1 if rrcaptions[i]==captions[i] else 0 for i, v in enumerate(gt_indexs)]
    
    # Step 1, Anchor Location
    images_stage1, prompts_stage1  = [], []
    for i, v in enumerate(gt_indexs):
        if conditions[i] == 1:
            images_stage1.append(ref_images[i][round])
            prompts_stage1.append(prompt1.format(cap=captions[i]))
    
    
    rss = batch_infer(llm, prompts_stage1, images_stage1, t=0.01)  
    rpl_ids  = [i for i, v in enumerate(gt_indexs) if conditions[i] == 1]

    for j, ids in enumerate(rpl_ids):
        if 'yes' in rss[j].lower():
            decisions[round][ids] = 1 
        else:
            decisions[round][ids] = 0

    rrpl_ids = []
    for j, ids in enumerate(rpl_ids):
        flg = 0
        for l in range(round):
            if decisions[l][ids] == 1:
                flg += 1  
        if decisions[round][ids] == 1 and round==0 and sims[ids] > xi: 
            gt_indexs[ids] = ref_indexs[ids][round]
            rrpl_ids.append(ids)

        if flg==0 and decisions[round][ids] == 1 and round>0 and sims[ids] <= xi:
            gt_indexs[ids] = ref_indexs[ids][round]
            ggt_indexs[ids] = ref_indexs[ids][round]
            rrpl_ids.append(ids)
    
    # step 2, Human-centered VQA    
    prompts_stage2 = [prompt2.format(cap=captions[v]) for v in rrpl_ids] 
    images_stage2  = [img_paths[gt_indexs[v]] for v in rrpl_ids]
    rs = batch_infer(llm, prompts_stage2, images_stage2, t=0.01)  
  
    rs = process_cap_(rs)
    for i, v in enumerate(rs): 
        rcaptions[rrpl_ids[i]] = v 
    prompts_stage3 = [prompt3.format(cap= [captions[v]] + rcaptions[v]) for v in rrpl_ids]
    rs = batch_infer_txt(llm, prompts_stage3, t=0.01)
    for i, v in enumerate(rs): 
        rrcaptions[rrpl_ids[i]] = v

def get_embeding(model,test_img_loader, test_txt_loader):
    if base_model != 'RDE': 
        qfeats_bge = []
        for pid, caption in test_txt_loader:
            caption = caption.cuda()
            with torch.no_grad():
                text_feat_bge = model.encode_text(caption).cpu()   
            qfeats_bge.append(text_feat_bge)  
        qfeats_bge = torch.cat(qfeats_bge, 0) 
        
        pfeats_bge = []
        for pid, img in test_img_loader:
            img = img.cuda()
            with torch.no_grad():
                img_feat = model.encode_image(img).cpu() 
            pfeats_bge.append(img_feat)
        pfeats_bge = torch.cat(pfeats_bge, 0) 
        return  F.normalize(qfeats_bge.cpu(), p=2, dim=1), \
                F.normalize(pfeats_bge.cpu(), p=2, dim=1)
    else:
        qfeats_bge, qfeats_tse = [], []
        for pid, caption, _ in test_txt_loader:
            caption = caption.cuda()
            with torch.no_grad():
                text_feat_bge = model.encode_text(caption).cpu()
                text_feat_tse = model.encode_text_tse(caption).cpu()     
            qfeats_bge.append(text_feat_bge)
            qfeats_tse.append(text_feat_tse) 

        qfeats_bge = torch.cat(qfeats_bge, 0) 
        qfeats_tse = torch.cat(qfeats_tse, 0)  

        # image
        pfeats_bge, pfeats_tse = [], []
        for pid, img in test_img_loader:
            img = img.cuda()
            with torch.no_grad():
                img_feat = model.encode_image(img).cpu() 
                img_feat1 = model.encode_image_tse(img).cpu() 
            pfeats_bge.append(img_feat)
            pfeats_tse.append(img_feat1)
            
        pfeats_bge = torch.cat(pfeats_bge, 0) 
        pfeats_tse = torch.cat(pfeats_tse, 0)  
            
        return  F.normalize(qfeats_bge.cpu(), p=2, dim=1), \
                F.normalize(qfeats_tse.cpu(), p=2, dim=1), \
                F.normalize(pfeats_bge.cpu(), p=2, dim=1), \
                F.normalize(pfeats_tse.cpu(), p=2, dim=1)
    
def get_cap_embeds(cap_rs_loader):
    model.eval()
    rqfeats_beg, rqfeats_tse = [], []
    for caption in cap_rs_loader:
        caption = caption.cuda()
        with torch.no_grad():
            text_feat = model.encode_text(caption).cpu()
            if base_model == 'RDE':
                rtext_feat = model.encode_text_tse(caption).cpu() 
            else: 
                rtext_feat = model.encode_text(caption).cpu() 
        rqfeats_beg.append(text_feat)
        rqfeats_tse.append(rtext_feat) 
    rqfeats_beg = F.normalize(torch.cat(rqfeats_beg, 0).cpu(), p=2, dim=1)
    rqfeats_tse =  F.normalize(torch.cat(rqfeats_tse, 0).cpu(), p=2, dim=1)
    return rqfeats_beg, rqfeats_tse
    
def eval(args, qids, pids, round, t = 0.8):
    # text 
    cap_rs_loader = DataLoader(TextPureDataset(rrcaptions, text_length=args.text_length), batch_size=batch_size, shuffle=False, num_workers=0)

    rqfeats_beg, rqfeats_tse = get_cap_embeds(cap_rs_loader)
    if base_model == 'RDE':
        sims_ =  (rqfeats_beg @ gfeats.t()  + rqfeats_tse @ vg_feats.t())/2  
    else:
        sims_ =  rqfeats_beg @ gfeats.t()

    global sims_base, global_sims
    for i, g in enumerate(gt_indexs):
        if g > -1: 
            tmp = sims_[i].clone()
            tmp[g] = 1
            sims_[i] = tmp
        else:
            sims_[i] = sims_base[i].clone()

    sims_ = sims_base * t + (1 - t)* sims_
    sims_dict = {
        'sims_base': sims_base,
        'sims_last': global_sims,
        'sims_now': sims_,
    }
    
    print_rs(sims_dict, qids, pids) 
    return sims_

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="ICL Args")
    parser.add_argument("--base_model", default='RDE', type=str)
    parser.add_argument("--root_dir", default='/home/qinyang/projects/data', type=str)
    parser.add_argument("--embed_model_path", default='/home/qinyang/projects/ACVPR/RDE_bak/final_logs_dyn/UFine6926/20241202_113207_RDE_TAL+sr0.3_tau0.015_margin0.1_n0.0+aug+aug_dual_fixed_dyn0.1', type=str)
    parser.add_argument("--source", default='UFine6926', type=str) # CUHK-PEDES ICFG-PEDES RSTPReid UFine6926
    parser.add_argument("--target", default='ICFG-PEDES', type=str)
    parser.add_argument("--model_dir", default='/home/qinyang/projects/ACVPR/LLaMA-Factory/A_qy/output/qwen2_vl_7b_lora_sft', type=str)    
    parser.add_argument("--tag", default='', type=str)
    parser.add_argument("--lambda", default=0.8, type=float)
    parser.add_argument("--xi", default=0.5, type=float)
    parser.add_argument("--round", default=5, type=int)
    paras = parser.parse_args()
    paras = vars(paras)
    base_model = paras['base_model']
    tttt = paras['lambda']
    xi = paras['xi']
    tag = f"{paras['source']}_{paras['target']}_xi{xi}_lam{tttt}_{base_model}{paras['tag']}"

    parser.add_argument("--config_file", default=f"{paras['embed_model_path']}/configs.yaml")
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
    args.training = False
    args.root_dir = paras['root_dir']
    args.dataset_name = paras['target']
    args.model_dir = paras['model_dir']

    base = f"{paras['embed_model_path']}/{tag}_{args.dataset_name}/"

    logger = setup_logger(base_model, save_dir=base, if_train=args.training) 
    logger.info(args)
    # save config
    with open(f'{base}config.json', 'w', encoding='utf-8') as f:
        json.dump(paras, f, ensure_ascii=False, indent=4)
    
    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    img_paths = test_img_loader.dataset.img_paths
    captions = test_txt_loader.dataset.captions
    rcaptions = [[] for _ in captions]
    rrcaptions = copy.deepcopy(captions)
    # rrcaptions = captions
    gt_indexs = [-1 for _ in captions] # 0 is no
    ggt_indexs = [-1 for _ in captions] # 0 is no
    qids, pids = test_txt_loader.dataset.caption_pids, test_img_loader.dataset.image_pids
    qids = torch.tensor(qids) 
    pids = torch.tensor(pids) 
    
    batch_size = 128
    img_loader = DataLoader(ImgDataset(img_paths),batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate)
    cap_loader = DataLoader(TxtDataset(captions), batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate) 
    
    model = get_pretrained_model(args)     

    if base_model == 'RDE':
        qfeats, vq_feats, gfeats, vg_feats = get_embeding(model,test_img_loader,test_txt_loader)
        sims_base =  (qfeats @ gfeats.t() + vq_feats @ vg_feats.t())/2
    else:
        qfeats, gfeats = get_embeding(model,test_img_loader,test_txt_loader)
        sims_base =  qfeats @ gfeats.t()
    global_sims = sims_base.clone() 
    # llm = MLLMs(model_dir='/home/qinyang/projects/ACVPR/2024-CVPR-RDE/LLMs/Qwen2-VL-7B-Instruct')
    # llm = MLLMs(model_dir='/home/qinyang/projects/ACVPR/LLaMA-Factory/A_qy/output/qwen2_vl_7b_lora_sft')
    llm = MLLMs(model_dir = args.model_dir)
    n_round = paras['round']
    decisions = np.array([[0 for _ in captions ] for rrr in range(n_round)])
    gt_indexs = [-1 for _ in captions] # 0 is no
    ggt_indexs = [-1 for _ in captions] # 0 is no
    start_time = time.time()

    for round in range(n_round):
        logger.info("=="*10 + f"Round-{round+1}")
        start_round_time = time.time()
        round_llm(args, round, llm)
        global_sims = eval(args, qids, pids, round, t=tttt) 
        end_round_time = time.time()
        logger.info(f"Round {round}, time: {(end_round_time-start_round_time):.0f}, all time time: {(end_round_time-start_time):.0f}")