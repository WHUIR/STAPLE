from torch.utils.data import Dataset, DataLoader
import pickle
import torch
from tqdm import tqdm
import os
import numpy as np
import random
import math
import copy
class DataSequential(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.length = 0
        self.data = None
        self.max_seq_length = args.max_seq_length
        self.max_item_tokens = 32
        self.max_token_length = args.max_token_length
        self.item_title_list = None
        self.candidate_index = []
        self.item_count = max(list(pickle.load(open(f"{args.data_path}/{args.dataset}/iid2asin.pkl", 'rb')).keys())) + 1
        self.args.item_count = self.item_count

        self.load_data()
        self.item_title_tokens = None
        self.target2seqidx = None
        self.target2seqidx_copy = None
        self.target2pop = None
        self.item2pop = None

        self.tokenize_item_titles()
        self.sample_valid(self.data)

        self.candi_item_attention_mask = None
        self.candi_item_input_ids = None
        self.generate_cate_items()
        self.generate_target2seqidx()
        self.generate_item2pop()



    def get_all_training_example(self):
        train_examples = []
        for item in range(self.length):
            item_inputs = self.generate_example_input(self.data[item], item)
            train_examples.append([item_inputs[3], item_inputs[2]])
        return train_examples, self.get_items_tokens()

    def generate_item2pop(self):
        review_datas = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/review_datas.pkl", 'rb'))
        item2popularity = [0] * self.item_count
        for user in tqdm(review_datas.keys(), desc='Splitting Train/Valid/Test'):
            for i in range(1, len(review_datas[user])):
                review = review_datas[user][i]
                if i < len(review_datas[user]) - 2:
                    item2popularity[review[0]] += 1
        self.item2pop = item2popularity
        self.args.item2pop = self.item2pop



    def generate_target2seqidx(self):
        if self.mode != 'train':
            return
        target2seqidx = [[] for _ in range(self.item_count)]
        for idx in range(len(self.data)):
            target_iid = self.data[idx][1]
            target2seqidx[target_iid].append(idx)
        self.target2seqidx = target2seqidx
        target2pop = [math.pow(len(x), self.args.sample_alpha) for x in target2seqidx]
        target2pop_sum = sum(target2pop)
        self.target2pop = [x / target2pop_sum for x in target2pop]
        self.target2seqidx_copy = copy.deepcopy(target2seqidx)


    def get_item_token(self, idx, sample=False):
        item_token = self.item_title_tokens[idx]
        if not sample or self.args.token_ratio == 1.0 or self.mode != 'train':
            return item_token
        sample_token = (torch.rand([len(item_token)]) < self.args.token_ratio).nonzero().squeeze(1)
        item_token = [item_token[t_idx] for t_idx in sample_token]
        return item_token

    def sample_valid(self, datas):
        if self.args.valid_ratio == 1 or self.mode != 'valid':
            return
        import random
        random.seed(42)
        sample_idx = random.sample(list(range(len(datas))), int(len(datas) * self.args.valid_ratio))
        sample_idx.sort()
        new_datas = []
        for idx in sample_idx:
            new_datas.append(datas[idx])
        self.length = len(new_datas)
        self.data = new_datas


    def __len__(self):
        return self.length

    def __getitem__(self, item):
        target_item = 0
        if self.mode == 'train' and self.args.sample_alpha != 1.0:
            while target_item == 0:
                if len(self.candidate_index) == 0:
                    self.candidate_index = np.random.choice(self.item_count, size=10000, p=self.target2pop).tolist()
                target_item = self.candidate_index.pop()
                if len(self.target2seqidx[target_item]) == 0:
                    target_item = 0
                else:
                    if len(self.target2seqidx_copy[target_item]) == 0:
                        self.target2seqidx_copy[target_item] = self.target2seqidx[target_item] + []
                        random.shuffle(self.target2seqidx_copy[target_item])
                    item = self.target2seqidx_copy[target_item].pop()

        example_input = self.generate_example_input(self.data[item], item)
        example_input.append(item)
        return example_input


    def load_data(self):
        review_datas = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/review_datas.pkl", 'rb'))
        train_data = []
        valid_data = []
        test_data = []

        for user in tqdm(review_datas.keys(), desc='Splitting Train/Valid/Test'):
            seq_iid_list = [review_datas[user][0][0]]
            seq_iid_cate_list = [review_datas[user][0][2]]
            for i in range(1, len(review_datas[user])):
                target_iid = review_datas[user][i][0]
                target_iid_cate = review_datas[user][i][2]
                if i < len(review_datas[user]) - 2:
                    train_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])
                elif i == len(review_datas[user]) - 2:
                    valid_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])
                elif i == len(review_datas[user]) - 1:
                    test_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])
                else:
                    raise NotImplementedError
                seq_iid_list = seq_iid_list + [review_datas[user][i][0]]
                seq_iid_cate_list = seq_iid_cate_list + [review_datas[user][i][2]]

                seq_iid_list = seq_iid_list[-self.max_seq_length:]
                seq_iid_cate_list = seq_iid_cate_list[-self.max_seq_length:]

        if self.mode == 'train':
            self.data = train_data
        elif self.mode == 'valid':
            self.data = valid_data
        elif self.mode == 'test':
            self.data = test_data
        else:
            raise NotImplementedError
        self.length = len(self.data)

    def generate_cate_items(self):
        candi_item_input_ids = []
        candi_item_attention_mask = []
        fp_tokens = self.max_item_tokens + 1
        for idx in range(self.item_count):
            candi_tokens = self.get_item_token(idx, True) + [self.tokenizer.eos_token_id]
            pad_len = fp_tokens - len(candi_tokens)
            candi_item_input_ids.append(candi_tokens + [0] * pad_len)
            candi_item_attention_mask.append((len(candi_tokens) * [1] + [0] * pad_len))
        self.candi_item_input_ids = candi_item_input_ids
        self.candi_item_attention_mask = candi_item_attention_mask

    def tokenize_item_titles(self):
        item_metas = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/meta_datas.pkl", 'rb'))
        iid2asin = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/iid2asin.pkl", 'rb'))
        id_prefix = 'id:'
        title_prefix = 'title:'
        item_title_list = ['None'] * self.item_count
        for iid, asin in iid2asin.items():
            item_title = item_metas[asin]['title'] if ('title' in item_metas[asin].keys() and item_metas[asin]['title']) else 'None'
            item_title = item_title.replace('&amp', '')
            item_title = id_prefix + ' ' + str(iid) + ' ' + title_prefix + ' ' + item_title + ', '
            item_title_list[iid] = item_title


        item_max_tokens = self.max_item_tokens
        item_title_tokens = []
        for start in tqdm(range(0, len(item_title_list), 32), desc='Tokenizing'):
            tokenized_text = self.tokenizer(item_title_list[start: start + 32],
                                            truncation=True,
                                            max_length=item_max_tokens,
                                            padding=False,
                                            add_special_tokens=False,
                                            return_tensors=None)

            item_title_tokens.extend(tokenized_text['input_ids'])
        self.item_title_tokens = item_title_tokens
        template1 = "Here is the visit history list of user: "
        template2 = " recommend next item "
        self.template1_ids = self.tokenizer.encode(template1, add_special_tokens=False, truncation=False)
        self.template2_ids = self.tokenizer.encode(template2, add_special_tokens=False, truncation=False)


    def generate_example_input(self, example, example_idx):
        seq_iid_list, target_iid = example[0], example[1]
        sequence_input_ids = []

        for seq_iid in seq_iid_list:
            seq_i_tokens = self.get_item_token(seq_iid, True)
            sequence_input_ids.extend(seq_i_tokens)

        sequence_input_ids = self.template1_ids + sequence_input_ids + self.template2_ids
        sequence_attention_mask = [1] * len(sequence_input_ids)

        sequence_input_ids = sequence_input_ids + [self.tokenizer.eos_token_id]
        sequence_attention_mask.append(1)

        # 第三部分，放置候选项
        if self.mode == 'train':
            negative_items = random.sample(range(1, self.item_count), self.args.train_nega_count)
            target_position = 0
        else:
            negative_items = [0] * self.args.nega_count
            target_position = random.randint(0, self.args.nega_count)
        negative_items = negative_items[0:target_position] + [target_iid] + negative_items[target_position:]
        negative_items_pop = [self.item2pop[x] for x in negative_items]

        if self.mode == 'train':
            candi_item_input_ids = [self.candi_item_input_ids[x] for x in negative_items]
            candi_item_attention_mask = [self.candi_item_attention_mask[x] for x in negative_items]
        else:
            candi_item_input_ids = [0] * len(negative_items)
            candi_item_attention_mask = [0] * len(negative_items)

        return [candi_item_input_ids, candi_item_attention_mask, sequence_attention_mask, sequence_input_ids, target_position, target_iid, negative_items, negative_items_pop]

    # 增加特殊符号

    def collate_fn(self, batch_data):
        # candi_item_input_ids, candi_item_attention_mask, sequence_attention_mask, sequence_input_ids, target_position, target_iid, negative_items

        item_input_ids = []
        item_attention_mask = []
        sequence_attention_mask = []
        sequence_input_ids = []
        target_position = []
        target_iid = []
        example_index = []
        negative_items = []
        negative_items_pop = []

        max_seq_length = max(len(x[2]) for x in batch_data)

        for example in batch_data:
            item_input_ids.extend(example[0])
            item_attention_mask.extend(example[1])

            seq_pad_len = max_seq_length - len(example[2])
            sequence_attention_mask.append(example[2] + seq_pad_len * [0])
            sequence_input_ids.append(example[3] + seq_pad_len * [0])
            target_position.append(example[4])
            target_iid.append(example[5])
            negative_items.append(example[6])
            negative_items_pop.append(example[7])
            example_index.append(example[-1])

        return {
            'item_input_ids': torch.LongTensor(item_input_ids),
            'item_attention_mask': torch.LongTensor(item_attention_mask),
            'sequence_attention_mask': torch.LongTensor(sequence_attention_mask),
            'sequence_input_ids': torch.LongTensor(sequence_input_ids),
            'target_position': torch.LongTensor(target_position),
            'target_iid': torch.LongTensor(target_iid),
            'example_index': torch.LongTensor(example_index),
            'negative_items': torch.LongTensor(negative_items),
            'negative_items_pop': torch.FloatTensor(negative_items_pop),
        }

    def get_items_tokens(self):
        item_ids = []
        item_attn = []
        fp_tokens = self.max_item_tokens + 1
        for iid in range(len(self.item_title_tokens)):
            item_tokens = self.get_item_token(iid) + [self.tokenizer.eos_token_id]
            pad_len = fp_tokens - len(item_tokens)
            item_ids.append(item_tokens + [0] * pad_len)
            item_attn.append(len(item_tokens) * [1] + pad_len * [0])
        return {'item_ids': torch.LongTensor(item_ids),
                'item_attn': torch.LongTensor(item_attn)}
