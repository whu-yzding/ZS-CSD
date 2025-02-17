import os
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from box import Box
from src.preprocess import *
from tqdm import tqdm
import logging
from sklearn.metrics import f1_score,accuracy_score
from src.common import *
from prettytable import PrettyTable
from model.Ours_SS_TCL import *

class Main:
    def __init__(self,args):
        # 读取 YAML 文件
        config = Box(yaml.load(open('src/config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        for k, v in vars(args).items():
            setattr(config, k, v)
        self.config = config
        self.formatted_time = config.time
        self.log_dir = f"./result/{self.config.model_type}_{self.formatted_time}/log/"
        self.pred_dir = f"./result/{self.config.model_type}_{self.formatted_time}/pred/"
        self.save_dir = f"./result/{self.config.model_type}_{self.formatted_time}/save/"
        self.pred_file = self.pred_dir + f'{self.config.seed}.csv'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        logging.basicConfig(
            filename = self.log_dir + f'{self.config.seed}.log',
            filemode='w',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        headers = ['cuda', 'tau', 'alpha', 'seed', 'bert_lr', 'other_lr', 'weight_decay', 'gru_hidden','gru_layer']
        data = [
            [config.cuda_index, config.tau, config.alpha, config.seed, config.bert_lr, config.other_lr, config.weight_decay, config.gru_hidden, config.gru_layer],
        ]
        table = PrettyTable(headers)
        for row in data:
            table.add_row(row)
        logging.info("\n" + str(table))
        logging.info(f'pid: {os.getpid()}')
        set_seed(self.config.seed)
        self.config.device = torch.device('cuda:{}'.format(self.config.cuda_index) if torch.cuda.is_available() else 'cpu')
        self.best_epoch = 0
        self.best_test_macro_f1 = 0.0
        self.best_test_f1_avg = 0.0
        self.best_test_acc = 0.0
        self.best_favor = 0.0
        self.best_against = 0.0
        self.best_neutral = 0.0
        
        
    def train_iter(self):
        self.model.train()
        self.global_step = 0
        running_loss = 0.0
        for data in tqdm(self.trainLoader):
            self.global_step += 1
            loss, _, _ = self.model(**data)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            running_loss += loss.item()
            print(f'[train] epoch:{self.global_epoch+1} step:{self.global_step} loss:{loss.item()}')
        train_loss = running_loss / len(self.trainLoader)
        return train_loss
            
    def evaluate_iter(self, dataLoader=None,mode=None):
        self.model.eval()
        val_loss = 0.0
        seq_preds = []
        seq_trues = []
        doc_id_lst = []
        if mode == 'dev':
            dataLoader = self.devLoader
        elif mode == 'test':
            dataLoader = self.testLoader
        with torch.no_grad():
            for data in tqdm(dataLoader):      
                loss, seq_output, labels = self.model(**data)
                val_loss += loss.item()
                seq_output = seq_output.detach().cpu().numpy()
                seq_output = np.argmax(seq_output, -1)
                labels = labels.detach().cpu().numpy()
                labels = labels.reshape(-1)
                seq_preds.extend(seq_output)
                seq_trues.extend(labels)
                doc_id_lst.extend(data['doc_id'])
        val_loss /= len(dataLoader)
        macro_f1, favor, against, neutral, f1_avg, acc = self.get_metrices(seq_trues,seq_preds)
        if mode == 'test':
            assert len(seq_trues) == len(seq_preds)
            result = {'doc_id': doc_id_lst,'true': seq_trues, 'pred': seq_preds}
            df = pd.DataFrame(result)
            df.to_csv(self.pred_file, index=False)
        return macro_f1, val_loss ,favor, against, neutral, f1_avg, acc
    
    def train(self):
        best_dev_f1 = 0.0
        for epoch in range(self.config.epoch_size):
            self.global_epoch = epoch
            train_loss = self.train_iter()
            dev_macro_f1 ,dev_loss ,_ ,_ ,_ ,dev_f1_avg ,dev_acc = self.evaluate_iter(mode='dev')
            logging.info(f'Epoch {epoch+1}/{self.config.epoch_size},Train Loss: {train_loss:.2f}, Val Loss: {dev_loss:.2f}, Val Macro F1: {100 * dev_macro_f1:.2f}, f1_avg: {100 * dev_f1_avg:.2f}, dev_acc: {100 * dev_acc:.2f}')
            
            if epoch+1 >= 1:
                if dev_f1_avg >= best_dev_f1:
                    best_dev_f1 = dev_f1_avg
                    # 测试
                    test_macro_f1 ,test_loss ,favor ,against ,neutral ,test_f1_avg ,test_acc = self.evaluate_iter(mode='test')
                    logging.info(f'Test Loss: {test_loss:.2f}, Test Macro F1: {100 * test_macro_f1:.2f}, f1_avg: {100 * test_f1_avg:.2f}, test_acc: {100 * test_acc:.2f}, Test Favor: {100 * favor:.2f}, Test Against: {100 * against:.2f}, Test Neutral: {100 * neutral:.2f}\n')
                    self.best_epoch = epoch+1
                    self.best_test_macro_f1 = test_macro_f1
                    self.best_test_f1_avg = test_f1_avg
                    self.best_test_acc = test_acc
                    self.best_favor = favor
                    self.best_against = against
                    self.best_neutral = neutral
                    torch.save(self.model.state_dict(), self.save_dir + 'best_model.pth')
                    
    def load_param(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        bert_lr = float(self.config.bert_lr)
        other_lr = float(self.config.other_lr)

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if 'bert' in n and not any(nd in n for nd in no_decay)], 'weight_decay': float(self.config.weight_decay), 'lr': bert_lr},
            {'params': [p for n, p in param_optimizer if 'bert' in n and any(nd in n for nd in no_decay)],'weight_decay': 0, 'lr': bert_lr},
            {'params': [p for n, p in param_optimizer if 'bert' not in n and not any(nd in n for nd in no_decay)], 'weight_decay': float(self.config.weight_decay), 'lr': other_lr},
            {'params': [p for n, p in param_optimizer if 'bert' not in n and any(nd in n for nd in no_decay)], 'weight_decay': 0, 'lr': other_lr}
        ]
        # 定义参数组
        self.optimizer = AdamW(optimizer_grouped_parameters, eps=float(self.config.adam_epsilon))
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_steps,
                                                         num_training_steps=self.config.epoch_size * self.trainLoader.__len__())
        
        
    def get_metrices(self, trues, preds):   
        f1_macro = f1_score(y_true=trues, y_pred=preds, average='macro')
        f1_per_class = f1_score(y_true=trues, y_pred=preds, average=None)
        favor = f1_per_class[0]
        against = f1_per_class[1]
        neutral = f1_per_class[2]
        f1_avg = (favor+against)/2
        accuracy = accuracy_score(y_true=trues,y_pred=preds)
        return f1_macro,favor,against,neutral,f1_avg,accuracy

    def save(self, save_path, save_name):
        torch.save(self.model.state_dict(), os.path.join(save_path, save_name))
        
    def forward(self):
        self.trainLoader, self.devLoader, self.testLoader = DataProcessor(self.config).get_data()
        self.model = Ours(self.config).to(self.config.device)
        self.load_param()
        logging.info('Start training...')
        self.train()
        logging.info('End training...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_lr', type=float, default=1e-5, help='learning rate for BERT layers')
    parser.add_argument('--other_lr', type=float, default=1e-5, help='learning rate for other layers')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='l2')
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--cuda_index', type=int, default=0, help='CUDA index')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA index')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', type=bool, default=False, help='random seed')
    parser.add_argument('--time', type=str, default='')
    parser.add_argument('--gru_layer', type=int, default=2)
    parser.add_argument('--gru_hidden', type=int, default=768)
    
    parser.add_argument('--model_type', type=str, default='Ours_SS_TCL')
    parser.add_argument('--plm', type=str, default='roberta', help='')
    parser.add_argument('--bert_dir', type=str, default='./plm/chinese-roberta-wwm-ext/')
    
    args = parser.parse_args()
    main = Main(args)
    main.forward()