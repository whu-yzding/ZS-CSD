from transformers import AutoTokenizer,BertTokenizer
import json
import numpy as np
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader,Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    

class DataProcessor():
    def __init__(self,config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_dir)
        self.config = config
    
    def read_data(self, mode):
        """
        Read a JSON file, tokenize using BERT, and realign the indices of the original elements according to the tokenization results.
        """
        if mode == 'train':
            file_path = self.config.train_path
        elif mode == 'dev':
            file_path = self.config.dev_path
        elif mode == 'test':
            file_path = self.config.test_path
        content = json.load(open(file_path, 'r', encoding='utf-8'))
        res = []
        for line in tqdm(content, desc='Processing dialogues for {}'.format(mode)):
            new_dialog = self.parse_dialogue(line, mode)
            res.append(new_dialog)
        return res
    
    def parse_dialogue(self, dialogue, mode):
        # get the list of sentences in the dialogue
        new_sentences = []
        new_label_sen = []
        target_indices = []
        target = dialogue["target"]
        speakers = dialogue["speakers"]
        for id,sen in enumerate(dialogue['sentences']):
            new_sen = f'[CLS]在话语“{sen}”中，用户{speakers[id]}对[SEP]{target}[SEP]的立场为[MASK][SEP]'
            if id == len(dialogue['sentences']) - 1:
                label_sen = f'[CLS]在话语“{sen}”中，用户{speakers[id]}对[SEP]{target}[SEP]的立场为{dialogue["label"]}[SEP]'
                label_sen_token = self.tokenizer.tokenize(label_sen)
                new_label_sen.append(label_sen_token)
            tokens = self.tokenizer.tokenize(new_sen)
            new_sentences.append(tokens)
            sep_indices = [i for i, token in enumerate(tokens) if token == '[SEP]']
            target_indices.append([sep_indices[0]+1,sep_indices[1]])
        dialogue['sentences'] = new_sentences
        dialogue['label_sen'] = new_label_sen
        dialogue['target_idx'] = target_indices
        return dialogue
    
    def transform2indices(self,data):
        res = []
        for document in data:
            sentences, speakers, label, target_idx, target, label_sen, doc_id = [document[w] for w in ['sentences', 'speakers', 'label','target_idx', 'target', 'label_sen', 'id']]
            input_ids = list(map(self.tokenizer.convert_tokens_to_ids, sentences))
            input_masks = [[1] * len(w) for w in input_ids]
            input_segments = [[0] * len(w) for w in input_ids]
            input_ids_label = list(map(self.tokenizer.convert_tokens_to_ids, label_sen))
            input_masks_label = [[1] * len(w) for w in input_ids_label]
            input_segments_label = [[0] * len(w) for w in input_ids_label]
            
            res.append((input_ids, input_masks, input_segments, speakers, label, target_idx, target, input_ids_label, input_masks_label ,input_segments_label, doc_id))
        return res
    
    def forward(self,mode):
        # modes = 'train valid test'
        data = self.read_data(mode)
        res = self.transform2indices(data)
        return res
    
    def collate_fn_new(self,batch):
        input_ids, input_masks, input_segments, speakers, label, target_idx, target, input_ids_label, input_masks_label ,input_segments_label, doc_id = zip(*batch)
        dialogue_length = list(map(len, input_ids))
        st = 0
        dia_idx = []
        for num in dialogue_length:
            dia_idx.append([st,st+num])
            st += num
        max_lens = max(len(w) for sublist in input_ids for w in sublist)
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for sublist in input_batch for w in sublist]
        input_ids, input_masks, input_segments = map(padding, [input_ids, input_masks, input_segments])
        max_lens_label = max(len(w) for sublist in input_ids_label for w in sublist)
        padding_label = lambda input_batch: [w + [0] * (max_lens_label - len(w)) for sublist in input_batch for w in sublist]
        input_ids_label, input_masks_label, input_segments_label = map(padding_label, [input_ids_label, input_masks_label, input_segments_label]) 
        res = {
            "input_ids":input_ids,
            "input_masks":input_masks,
            "input_segments":input_segments,
            "speakers":speakers,
            "label":label,
            "dia_idx":dia_idx,
            "target_idx":target_idx,
            "target": target,
            "input_ids_label": input_ids_label,
            "input_masks_label": input_masks_label,
            "input_segments_label": input_segments_label,
            "doc_id": doc_id
        }
        nocuda = ["speakers","dia_idx", "target_idx", "target", "doc_id"]
        res = {k: v if k in nocuda else torch.tensor(v).to(self.config.device) for k, v in res.items()}
        return res
    
    def get_data(self):
        if self.config.debug:
            train_dataset = MyDataset(self.forward('train'))[:100]
            dev_dataset = MyDataset(self.forward('dev'))[:100]
            test_dataset = MyDataset(self.forward('test'))[:100]
        else:
            train_dataset = MyDataset(self.forward('train'))
            dev_dataset = MyDataset(self.forward('dev'))
            test_dataset = MyDataset(self.forward('test'))
        train_loader = DataLoader(train_dataset, batch_size=self.config.batchsize, shuffle=True,collate_fn=self.collate_fn_new)
        dev_loader = DataLoader(dev_dataset, batch_size=self.config.batchsize, shuffle=True,collate_fn=self.collate_fn_new)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batchsize, shuffle=True,collate_fn=self.collate_fn_new)
        return train_loader, dev_loader, test_loader

    


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--bert_dir', type=str, default='./plm/chinese-roberta-wwm-ext/')
#     parser.add_argument('--test_path', type=str, default='./data/dev_data.json')
#     args = parser.parse_args()
#     Processor = DataProcessor(args)
#     test_dataset = MyDataset(Processor.forward('test'))
#     # print(test_dataset[0])
#     test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True,collate_fn=collate_fn_new)
#     for batch in test_loader:
#         inputs = batch['input_ids']
#         input_masks = batch['input_masks']
