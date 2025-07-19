from torch.utils.data import Dataset
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import numpy as np
import os
import nltk

def inject_noisy_correspondence(dataset, noisy_rate, noisy_file =None):
    if noisy_rate == 0:
        return dataset, np.array([1 for _ in dataset])
    
    logger = logging.getLogger("RDE.dataset")
    nums = len(dataset)
    dataset_copy = dataset.copy()
    captions  = [i[3] for i in dataset_copy]
    images    = [i[2] for i in dataset_copy]
    image_ids = [i[1] for i in dataset_copy]
    pids      = [i[0] for i in dataset_copy]

    noisy_inx = np.arange(nums)
    if noisy_rate > 0:
        print(noisy_file)
        random.seed(123)
        if os.path.exists(noisy_file):
            logger.info('=> Load noisy index from {}'.format(noisy_file))
            noisy_inx = np.load(noisy_file)
        else:
            inx = np.arange(nums)
            np.random.shuffle(inx)
            c_noisy_inx = inx[0: int(noisy_rate * nums)]
            shuffle_noisy_inx = np.array(c_noisy_inx)
            np.random.shuffle(shuffle_noisy_inx)
            noisy_inx[c_noisy_inx] = shuffle_noisy_inx
            np.save(noisy_file, noisy_inx)

    real_correspondeces = []
    for i in range(nums):
        if noisy_inx[i]== i:
            real_correspondeces.append(1)
        else:
            real_correspondeces.append(0)
        # pid, real_pid, image_id, image_path, text
        tmp = (pids[i],image_ids[i],images[i],captions[noisy_inx[i]])
        dataset[i] = tmp
    logger.info(real_correspondeces[0:10])
    logger.info('=>Noisy rate: {},  clean pairs: {}, noisy pairs: {}, total pairs: {}'.format(noisy_rate, np.sum(real_correspondeces),nums-np.sum(real_correspondeces), nums))

    return dataset, np.array(real_correspondeces)

class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("RDE.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result
 

class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform
        
        # add by cross
        from utils.iotools import read_json
        import os.path as op
        self.anno_path = '/home/qinyang/projects/ACVPR/RDE_bak/data/data_1110/r_icfg_1110_dec_1110.json'
        test_annos = [] 
        for anno in read_json(self.anno_path): 
            if anno['split'] == 'test':
                test_annos.append(anno)
        
        # print("old dataset len(images)", len(self.img_paths)) 
        # self.img_dir = '/home/qinyang/projects/data/ICFG-PEDES/imgs'
        # add_image_pids = [] 
        # max_id = max(self.image_pids) + 1
        # for anno in test_annos:
        #     pid = int(anno['id']) + max_id 
        #     img_path = op.join(self.img_dir, anno['file_path'])
        #     self.img_paths.append(img_path)
        #     self.image_pids.append(pid)
        # print("new dataset len(images)", len(self.img_paths))

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        
 
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 dec_captions = None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.dec_captions = dec_captions
  
    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]
        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        return pid, caption, index

class TextPureDataset(Dataset):
    def __init__(self, 
                 captions, 
                 text_length: int = 77,
                 truncate: bool = True): 
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer() 
  
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index): 
        caption = tokenize(self.captions[index], tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        return caption
    
class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset, args,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.txt_aug = args.txt_aug
        self.img_aug = args.img_aug
       
        self.dataset, self.real_correspondences = inject_noisy_correspondence(dataset,args.noisy_rate,args.noisy_file)
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption, dec_caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        def get_dev_cap(dec_caps):
            if type(dec_caps) is str:
                return dec_caps
            random.shuffle(dec_caps)
            tmp = ""
            for i, r in enumerate(dec_caps):
                rrr = random.choice(r)  
                tmp += rrr + " " 
            return tmp[:-1]

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        rcaption_tokens = tokenize(get_dev_cap(dec_caption), tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        if self.txt_aug: 
            caption_tokens = self.txt_data_aug(caption_tokens.cpu().numpy()) 
            rcaption_tokens = self.txt_data_aug(rcaption_tokens.cpu().numpy()) 
        
        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'rcaption_ids':rcaption_tokens,
            'index':index,
        }

        return ret

    def txt_data_aug(self, tokens, levels=None):
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        new_tokens = np.zeros_like(tokens)
        aug_tokens = []
        for i, token in enumerate(tokens): 
            if 0 < token < 49405:
                if levels is not None:
                    p = 0.1 * levels[i]
                else:
                    p = 0.2
                prob = random.random()
                # mask token with 15% probability
                if prob < p:
                    prob /= p
                    # 50% randomly change token to mask token
                    if prob < 0.6:
                        aug_tokens.append(mask) 
                    # 20% randomly change token to random token
                    elif prob < 0.8:
                        aug_tokens.append(random.choice(token_range)) # -> rest 10% randomly keep current token
                    else:
                        None # 20% randomly remove
                else:
                    # no masking token (will be ignored by loss function later)
                    aug_tokens.append(tokens[i])
            else:
                aug_tokens.append(tokens[i])
        new_tokens[0:len(aug_tokens)] = np.array(aug_tokens)
        return torch.tensor(new_tokens)