import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
from src.common import *

class Ours(nn.Module):
    def __init__(self, config):
        super(Ours, self).__init__()
        self.config = config
        self.alpha = config.alpha
        self.beta = config.beta
        self.bert = AutoModel.from_pretrained(config.bert_dir)
        self.bert_config = self.bert.config
        self.gru = nn.GRU(input_size=768, hidden_size=config.gru_hidden, num_layers=config.gru_layer, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(config.gru_hidden, config.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.SSE = SSE()
        
    
    def forward(self,**kwargs):
        input_ids, input_masks, input_segments, speakers, label, dia_idx, target_idx, targets, input_ids_label, input_masks_label, input_segments_label = [kwargs[w] for w in ['input_ids', 'input_masks', 'input_segments','speakers','label','dia_idx','target_idx','target','input_ids_label', 'input_masks_label', 'input_segments_label']]
        out = self.bert(input_ids=input_ids, attention_mask=input_masks, token_type_ids=input_segments).last_hidden_state
        H_final = []
        stance = []
        dia_id = 0
        for st,ed in dia_idx:
            h = out[st:ed,-2,:]
            o, _ = self.gru(h.unsqueeze(0))
            o = o.squeeze(0)
            v = self.SSE(o, speakers[dia_id])
            H_final.append(v)
            stance.append(v[-1])
            dia_id += 1
        stance = torch.stack(stance)
        stance = self.fc(stance)
        ce_loss = self.criterion(stance,label)
        target_contrastive_loss = target_CL(H_final,targets,self.config)
        loss = ce_loss + self.alpha * target_contrastive_loss
        
        return loss, stance, label