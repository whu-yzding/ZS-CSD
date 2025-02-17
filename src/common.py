import os
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def map_sequence(input_list):
    num_to_order = {}
    order = 0
    result = []
    for num in input_list:
        if num not in num_to_order:
            num_to_order[num] = order
            order += 1
        result.append(num_to_order[num])
    return result

    
class SSE(nn.Module):
    def __init__(self):
        super(SSE, self).__init__()
        self.linear_intra = nn.Linear(768 * 2, 768)
        self.linear_inter = nn.Linear(768,768)
        self.attention_intra = Attention(768)
        self.attention_inter = Attention(768)
    def forward(self, utterances, speaker):
        speaker = map_sequence(speaker)
        V_lst = []
        for i in range(len(speaker)):
            if i == 0:
                v = utterances[i]
                V_lst.append(v)
            else:
                tmp = -1
                for j in range(i):
                    if speaker[j] == speaker[i]:
                        tmp = j
                if tmp == -1:
                    v = utterances[i]
                    V_lst.append(v)
                else:
                    vh_concat = torch.cat((V_lst[tmp], utterances[i]), dim=-1)
                    q_intra = self.linear_intra(vh_concat)
                    c = utterances[:i+1]
                    v_intra = self.attention_intra(q_intra, c, c)
                    q_inter = self.linear_inter(utterances[i])
                    k = torch.stack(V_lst[tmp:i])
                    v_inter = self.attention_inter(q_inter, k, k)
                    v = v_intra + v_inter
                    V_lst.append(v)
        return torch.stack(V_lst)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, query, keys, values):
        q = self.query_layer(query)
        k = self.key_layer(keys)
        v = self.value_layer(values)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attention_weights = self.softmax(scores)
        output = torch.matmul(attention_weights, v)
        return output

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False 
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def compute_target_prototypes(H_lst, new_target, num_classes):
    prototypes = []
    vectors = []
    labels = []
    for c in range(num_classes):
        prototype = []
        for i in range(len(new_target)):
            if new_target[i] == c:
                prototype.extend([utt for utt in H_lst[i]])
                vectors.extend([utt for utt in H_lst[i]])
                labels.extend([c for utt in H_lst[i]])
        prototype = torch.stack(prototype)
        prototype = prototype.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes),torch.stack(vectors),torch.tensor(labels)

def target_CL(H_lst, target, config):
    tar_dic = {}
    tmp = 0
    for t in target:
        if t not in tar_dic.keys():
            tar_dic[t] = tmp
            tmp += 1
    new_target = [tar_dic[t] for t in target]
    num_classes = tmp
    prototypes,vectors,labels = compute_target_prototypes(H_lst, new_target, num_classes)
    vectors = F.normalize(vectors, p=2, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    similarities = torch.mm(vectors, prototypes.T)
    similarities /= config.tau
    labels_one_hot = F.one_hot(labels, num_classes).float().to(config.device)
    pos_sim = (similarities * labels_one_hot).sum(dim=1)
    loss = -torch.log(torch.exp(pos_sim) / torch.exp(similarities).sum(dim=1)).mean()
    return loss