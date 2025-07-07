import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
from src.common import map_sequence, target_CL

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, query, keys, values):
        # query: (B, D) or (D,)
        # keys/values: (L, D)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        q = self.query_proj(query)         # (B, D)
        k = self.key_proj(keys)            # (L, D)
        v = self.value_proj(values)        # (L, D)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, L)
        attn_weights = torch.softmax(attn_scores, dim=-1)                # (B, L)
        output = torch.matmul(attn_weights, v)                           # (B, D)
        return output.squeeze(0) if output.size(0) == 1 else output

class SSE(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.linear_intra = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear_inter = nn.Linear(hidden_dim, hidden_dim)
        self.attention_intra = Attention(hidden_dim)
        self.attention_inter = Attention(hidden_dim)

    def forward(self, utterances, speakers):
        # utterances: (L, D)  speakers: list or tensor of L
        device = utterances.device
        speakers = torch.tensor(map_sequence(speakers), device=device)
        V_lst = []
        last_speaker_idx = dict()
        for i in range(len(speakers)):
            speaker_id = speakers[i].item()
            if speaker_id not in last_speaker_idx:
                V_lst.append(utterances[i])
            else:
                prev_idx = last_speaker_idx[speaker_id]
                vh_concat = torch.cat((V_lst[prev_idx], utterances[i]), dim=-1)
                q_intra = self.linear_intra(vh_concat)
                c = utterances[:i+1]
                v_intra = self.attention_intra(q_intra, c, c)

                q_inter = self.linear_inter(utterances[i])
                k = torch.stack([V_lst[j] for j in range(prev_idx, i)]) if i > prev_idx else utterances[i].unsqueeze(0)
                v_inter = self.attention_inter(q_inter, k, k) if len(k) > 0 else torch.zeros_like(q_inter)

                V_lst.append(v_intra + v_inter)
            last_speaker_idx[speaker_id] = i
        return torch.stack(V_lst)

class SITCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.alpha = config.alpha
        self.bert = AutoModel.from_pretrained(config.bert_dir)
        self.gru = nn.GRU(input_size=768, hidden_size=config.gru_hidden, num_layers=config.gru_layer, batch_first=True)
        self.fc = nn.Linear(config.gru_hidden, config.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.SSE = SSE(hidden_dim=config.gru_hidden)

    def forward(self, **kwargs):
        input_ids = kwargs['input_ids']
        input_masks = kwargs['input_masks']
        input_segments = kwargs['input_segments']
        speakers = kwargs['speakers']
        label = kwargs['label']
        dia_idx = kwargs['dia_idx']
        targets = kwargs['target']

        out = self.bert(input_ids=input_ids, attention_mask=input_masks, token_type_ids=input_segments).last_hidden_state

        H_final = []
        stance = []
        for dia_id, (st, ed) in enumerate(dia_idx):
            h = out[st:ed, -2, :]
            o, _ = self.gru(h.unsqueeze(0))      # (1, L, D)
            o = o.squeeze(0)                     # (L, D)
            v = self.SSE(o, speakers[dia_id])
            H_final.append(v)
            stance.append(v[-1])
        stance = torch.stack(stance)             # (B, D)
        logits = self.fc(stance)                 # (B, num_classes)
        ce_loss = self.criterion(logits, label)
        target_contrastive_loss = target_CL(H_final, targets, self.config)
        loss = ce_loss + self.alpha * target_contrastive_loss
        return loss, logits, label
