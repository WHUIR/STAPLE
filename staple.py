import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch import autocast
from utils import info


class STAPLE(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args

        from transformers import OPTModel
        self.llm = OPTModel.from_pretrained(args.root_path + args.backbone)
        self.llm_debias = OPTModel.from_pretrained(args.root_path + args.debias_backbone)

        self.en_de_bias_debias = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size),
            nn.LeakyReLU(),
        )

        self.en_de_bias = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size),
            nn.LeakyReLU(),
        )

        self.freeze_stage_params()

        self.item_embs = None
        self.item_embs_debias = None

    def freeze_stage_params(self):
        if self.args.train_stage == 1:
            for param in self.llm.parameters():
                param.requires_grad = True
            for param in self.en_de_bias.parameters():
                param.requires_grad = True

        if self.args.train_stage == 2:
            if not os.path.isfile(self.args.output + f"{self.args.dataset}-1.pth"):
                raise NotImplementedError('Missing stage1 checkpoint!')

            weights_stage1 = torch.load(self.args.output + f"{self.args.dataset}-1.pth", map_location=next(self.llm_debias.parameters()).device)
            info(self.load_state_dict(weights_stage1, strict=False))
            for param in self.llm.parameters():
                param.requires_grad = False
            for param in self.en_de_bias.parameters():
                param.requires_grad = True

        if self.args.train_stage == 3:
            if not os.path.isfile(self.args.output + f"{self.args.dataset}-1.pth"):
                raise NotImplementedError('Missing stage1 checkpoint!')
            if not os.path.isfile(self.args.output + f"{self.args.dataset}-2.pth"):
                raise NotImplementedError('Missing stage2 checkpoint!')

            weights_stage1 = torch.load(self.args.output + f"{self.args.dataset}-1.pth", map_location=next(self.llm_debias.parameters()).device)
            weights_stage2 = torch.load(self.args.output + f"{self.args.dataset}-2.pth", map_location=next(self.llm_debias.parameters()).device)
            weights_stage1.update(weights_stage2)
            weights = weights_stage1
            new_weights = {}
            for key in weights.keys():
                if 'llm' in key:
                    new_weights[key.replace('llm', 'llm_debias')] = weights[key]
                if 'en_de_bias' in key:
                    new_weights[key.replace('en_de_bias', 'en_de_bias_debias')] = weights[key]
            info(self.load_state_dict(new_weights, strict=False))

            for param in self.llm.parameters():
                param.requires_grad = True
            for param in self.en_de_bias.parameters():
                param.requires_grad = True

            for param in self.llm_debias.parameters():
                param.requires_grad = False
            for param in self.en_de_bias_debias.parameters():
                param.requires_grad = False


        for param in self.llm_debias.parameters():
            param.requires_grad = False

    def trainable2float(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                info(f"Trainable Parameter:{name}")
                param.data = param.data.float()

    def get_embedding(self, input_ids, attention_mask):
        llm_output = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        if self.args.train_stage != 3:
            return self.en_de_bias(self.gather_indexes(llm_output.last_hidden_state, attention_mask.sum(dim=-1) - 1))
        else:
            return self.gather_indexes(llm_output.last_hidden_state, attention_mask.sum(dim=-1) - 1)


    def get_debias_embedding(self, input_ids, attention_mask):
        llm_output = self.llm_debias(input_ids=input_ids, attention_mask=attention_mask)
        return self.en_de_bias_debias(self.gather_indexes(llm_output.last_hidden_state, attention_mask.sum(dim=-1) - 1))


    def reshape_item_cls(self, item_cls, negative_items):
        item_cls = item_cls.view(-1, item_cls.size()[-1])
        if self.args.nega_strategy == 'random':
            item_cls = item_cls.view(-1, self.args.train_nega_count + 1, item_cls.size()[-1])
            item_target_cls = item_cls[:, 0].unsqueeze(1)    # bs * 1 * 786
            item_negative_cls = item_cls[:, 1:].reshape(1, -1, item_target_cls.size(2)).repeat(item_target_cls.size(0), 1, 1)  # bs * (bs * 10) * 786
            item_cls = torch.cat([item_target_cls, item_negative_cls], dim=1)
            target_position = torch.zeros([item_cls.size(0)], device=item_cls.device).long()

            negative_items_target = negative_items[:, 0].unsqueeze(1)  # bs * 1
            negative_items_others = negative_items[:, 1:].reshape(1, -1).repeat(item_target_cls.size(0), 1)# bs * (bs * negs)
            negative_items = torch.cat([negative_items_target, negative_items_others], dim=1)

        elif self.args.nega_strategy == 'random+inbatch':
            batch_size = item_cls.size(0) // (self.args.train_nega_count + 1)
            item_cls = item_cls.unsqueeze(0).repeat(batch_size, 1, 1)
            target_position = torch.arange(
                item_cls.size(0),
                device=item_cls.device,
                dtype=torch.long
            ) * (self.args.train_nega_count + 1)
            negative_items = negative_items.reshape(1, -1).repeat(batch_size, 1)
        else:
            raise NotImplementedError

        return item_cls, target_position, negative_items

    def forward(self, inputs):
        seq_cls = self.get_embedding(input_ids=inputs['sequence_input_ids'], attention_mask=inputs['sequence_attention_mask'])
        item_cls = self.get_embedding(input_ids=inputs['item_input_ids'], attention_mask=inputs['item_attention_mask']) # (batch * 11) * 768
        item_cls, target_position, negative_items = self.reshape_item_cls(item_cls, inputs['negative_items'])

        if self.args.train_stage == 3:
            with torch.no_grad():
                item_cls_teacher = self.get_debias_embedding(input_ids=inputs['item_input_ids'], attention_mask=inputs['item_attention_mask'])
                seq_cls_teacher = self.get_debias_embedding(input_ids=inputs['sequence_input_ids'], attention_mask=inputs['sequence_attention_mask'])
            item_cls_teacher, _, _ = self.reshape_item_cls(item_cls_teacher, inputs['negative_items'])

            item_cls_teacher = item_cls_teacher.float()
            seq_cls_teacher = seq_cls_teacher.float().unsqueeze(-1)
            scores_teacher = torch.bmm(item_cls_teacher, seq_cls_teacher).squeeze(-1)
        else:
            scores_teacher = None

        item_cls = item_cls.float()
        seq_cls = seq_cls.float().unsqueeze(-1)
        if self.args.scaled_dot:
            scores_student = torch.bmm(item_cls, seq_cls).squeeze(-1) / math.sqrt(item_cls.size()[-1])
        else:
            scores_student = torch.bmm(item_cls, seq_cls).squeeze(-1)

        rec_loss = F.cross_entropy(scores_student, target_position)

        if self.args.train_stage == 3 and self.args.debias_alpha != 0:
            rec_loss_debias = self.cal_debias_loss(scores_student, scores_teacher, target_position, negative_items)
        else:
            rec_loss_debias = torch.zeros_like(rec_loss)

        return [rec_loss + rec_loss_debias * self.args.debias_alpha, rec_loss, rec_loss_debias]

    def cal_debias_loss(self, scores_student, scores_teacher, target_item, negative_items):
        if self.args.distill_type == 1:  # pair-wise loss
            scores_student_target = torch.gather(scores_student, dim=-1, index=target_item.unsqueeze(-1))
            scores_teacher_target = torch.gather(scores_teacher, dim=-1, index=target_item.unsqueeze(-1))

            scores_debias_target_positive = scores_teacher_target > scores_teacher  # target是正样本
            scores_debias_target_negative = scores_teacher_target < scores_teacher  # target是负样本

            bpr_loss_positive = -F.logsigmoid(scores_student_target - scores_student) * scores_debias_target_positive
            bpr_loss_negative = -F.logsigmoid(scores_student - scores_student_target) * scores_debias_target_negative

            if scores_debias_target_positive.sum() > 0:
                bpr_loss_positive = bpr_loss_positive.sum() / scores_debias_target_positive.sum()
            else:
                bpr_loss_positive = 0
            if scores_debias_target_negative.sum() > 0:
                item2pop = torch.tensor(self.args.item2pop, device=scores_student.device)[: self.args.item_count]
                pop_weight = item2pop[negative_items.view(-1)].view(negative_items.size())
                pop_weight = 1 / (pop_weight + 5)
                pop_weight = pop_weight * scores_debias_target_negative
                pop_weight = pop_weight / pop_weight.sum() * scores_debias_target_negative.sum()
                bpr_loss_negative = bpr_loss_negative * pop_weight

                bpr_loss_negative = bpr_loss_negative.sum() / scores_debias_target_negative.sum()
            else:
                bpr_loss_negative = 0
            return (bpr_loss_positive + bpr_loss_negative) / 2
        elif self.args.distill_type == 2:  # hard distillation
            teacher_label = scores_teacher.max(dim=-1)[1]
            distill_loss = F.cross_entropy(scores_student, teacher_label)
            return distill_loss.half()
        elif self.args.distill_type == 3:  # soft distillation
            distribution_student = F.log_softmax(scores_student, dim=-1)
            distribution_teacher = F.softmax(scores_teacher, dim=-1)
            distill_loss = F.kl_div(distribution_student, distribution_teacher, reduction='batchmean')
            return distill_loss.half()


    def valid_step(self, inputs):
        seq_cls = self.get_embedding(input_ids=inputs['sequence_input_ids'], attention_mask=inputs['sequence_attention_mask'])
        item_cls = self.item_embs[inputs['negative_items']].to(seq_cls.device)
        with autocast(device_type='cuda', enabled=False):
            seq_cls = seq_cls.float().unsqueeze(-1)
            scores = seq_cls.squeeze(-1) @ self.item_embs.float().t()
            label = inputs['target_iid']
        return scores, label

    @torch.no_grad()
    def generate_embs(self, item_tokens):
        del self.item_embs
        torch.cuda.empty_cache()
        info(f"GPU:{self.args.gpu} Generating Emebedding")
        item_ids = item_tokens['item_ids']
        item_attn = item_tokens['item_attn']
        device = next(self.parameters()).device

        item_embs = []
        batch_size = 128
        for start_idx in range(0, item_ids.size()[0], batch_size):
            batch_item_ids = item_ids[start_idx: start_idx + batch_size].to(device)
            batch_item_attn = item_attn[start_idx: start_idx + batch_size].to(device)
            batch_item_embs = self.get_embedding(input_ids=batch_item_ids, attention_mask=batch_item_attn)
            item_embs.append(batch_item_embs.cpu())
            torch.cuda.empty_cache()
        self.item_embs = torch.cat([x.to(device) for x in item_embs], dim=0)
        assert self.item_embs.size()[0] == item_ids.size()[0]

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
