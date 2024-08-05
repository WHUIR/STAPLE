import os
import pickle
import torch.multiprocessing as mp
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from dataloader import get_dataloader
from utils import LossMeter
from trainer_base import TrainerBase
from param import parse_args
from staple import STAPLE
import time
from torch.cuda.amp import GradScaler
from torch import autocast
from metrics import cal_recall, cal_ndcg, cal_gini, cal_cratio
from tqdm import tqdm
from utils import info

# The Trainer inherits TrainerBase in trainer_base.py
class Trainer(TrainerBase):
    def __init__(self, args, tokenizer, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        # config = self.create_config()
        self.model = STAPLE(args, tokenizer)
        self.tokenizer = tokenizer

        # GPU Options
        info(f'Model Launching at GPU {self.args.gpu}')
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True)
        if args.load:
            self.load(args.load)

        self.loss_names = ['total_loss', 'rec_loss', 'rec_debias']
        self.best_valid_result = 0
        self.early_stop_step = 0
        self.print_trainable_parameters()
        self.start_epoch = args.start_epoch
        if self.start_epoch != 0:
            info(f"Load model from epoch {self.start_epoch}!")
            raise NotImplementedError

    def train(self):

        if self.args.distributed:
            dist.barrier()
        if self.args.valid_first:
            if not self.args.skip_valid:
                self.valid_epoch(-1)
            if self.args.distributed:
                dist.barrier()
            if self.args.test_only:
                self.valid_epoch(-1, mode='test')
                return

        global_step = 1
        scaler = GradScaler()
        result = {'exit': False}

        for epoch in range(self.args.epoch):
            if self.start_epoch != 0:
                epoch += self.start_epoch
            epoch_start_time = time.time()
            # Train
            self.model.train()
            loss_meters = [LossMeter(100) for _ in range(len(self.loss_names))]
            loader_length = len(self.train_loader)
            logger_batch = (loader_length//100) + 1
            for step_i, batch in enumerate(self.train_loader):
                self.transfer_device(batch)
                with autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.fp16):
                    losses = self.model(batch)
                    loss = losses[0] / self.args.gradient_accumulation_steps
                scaler.scale(loss).backward()

                for i in range(len(loss_meters)):
                    loss_meters[i].update(losses[i].detach())

                if (step_i + 1) % self.args.gradient_accumulation_steps == 0:
                    global_step += 1
                    if self.args.clip_grad_norm > 0:
                        scaler.unscale_(optimizer=self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    scaler.step(self.optim)
                    scaler.update()
                    self.optim.zero_grad()
                    if self.lr_scheduler:
                        self.lr_scheduler.step()

                if step_i % self.args.gradient_accumulation_steps == 0:
                    remain_year, remain_min, remain_sec = self.remain_time(epoch_start_time, step_i, loader_length)
                    log_str = f"Global Step:{global_step} | Train Epoch {epoch} | Step:{step_i} / {loader_length} | Remain Time:{remain_year}:{remain_min}:{remain_sec} | "
                    for i in range(len(loss_meters)):
                        log_str += f'{self.loss_names[i]}:{loss_meters[i].val:.3f} | '
                    if self.args.gpu == 0 and ((step_i % logger_batch) == 0):
                        info(log_str)

                if self.args.save_by_step != 0 and global_step % self.args.save_by_step == 0 and step_i % self.args.gradient_accumulation_steps==0:
                    result = self.valid_epoch(global_step)
                    if result['save'] and self.args.gpu == 0:
                        info(f"Save model at global step at {global_step}")
                        self.save(self.args.output + f'{self.args.dataset}-{self.args.train_stage}.pth')
                    self.model.train()
                    if self.args.distributed:
                        dist.barrier()

            if self.args.distributed:
                dist.barrier()

            # Skip validation
            if self.args.save_by_step == 0:
                if self.args.skip_valid:
                    info("Skip Valid Save model At Epoch%02d" % (epoch + 1))
                    if self.args.gpu == 0:
                        self.save(self.args.output + f'{self.args.dataset}-{self.args.train_stage}.pth')
                    result = {'exit': False}
                else:
                    info(f"Evaluate At GPU-{self.args.gpu}!!!")
                    result = self.valid_epoch(epoch)
                    if result['save'] and self.args.gpu == 0:
                        info("Save model At Epoch%02d" % (epoch + 1))
                        self.save(self.args.output + f'{self.args.dataset}-{self.args.train_stage}.pth')

            if self.args.distributed:
                dist.barrier()

            if result['exit']:
                info(f"Test At GPU-{self.args.gpu}!!!")
                self.load(self.args.output + f'{self.args.dataset}-{self.args.train_stage}.pth')
                self.valid_epoch('Test', mode='test')
                if self.args.distributed:
                    dist.barrier()
                return

    @torch.no_grad()
    def valid_epoch(self, epoch, mode='valid'):
        dataloader = self.val_loader if mode == 'valid' else self.test_loader
        self.model.eval()
        with autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.fp16):
            if self.args.distributed:
                self.model.module.generate_embs(dataloader.dataset.get_items_tokens())
                dist.barrier()
            else:
                self.model.generate_embs(dataloader.dataset.get_items_tokens())
        loader_length = len(dataloader)
        logger_batch = (loader_length//10) + 1
        predict_score = []
        label = []
        example_index = []
        candidate_items = []
        real_label = []

        for batch_idx, batch_data in enumerate(dataloader):
            self.transfer_device(batch_data)
            if (self.args.gpu == 0) and ((batch_idx % logger_batch) == 0):
                info(f"Local Rank{self.args.gpu}-Evaluation:{batch_idx}/{loader_length}")
            with autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.fp16):
                if self.args.distributed:
                    scores, bs_label = self.model.module.valid_step(batch_data)
                else:
                    scores, bs_label = self.model.valid_step(batch_data)
            example_index.append(batch_data['example_index'])
            label.append(bs_label)
            predict_score.append(scores)
            candidate_items.append(batch_data['negative_items'])
            real_label.append(batch_data['target_iid'])

        label = torch.cat(label, dim=0)
        example_index = torch.cat(example_index, dim=0)
        predict_score = torch.cat(predict_score, dim=0).to(example_index.device)
        candidate_items = torch.cat(candidate_items, dim=0).to(example_index.device)
        real_label = torch.cat(real_label, dim=0).to(example_index.device)

        if self.args.distributed:
            all_predict_score = [torch.zeros_like(predict_score) for _ in range(self.args.num_gpus)]
            dist.all_gather(all_predict_score, predict_score.contiguous())

            all_label = [torch.zeros_like(label) for _ in range(self.args.num_gpus)]
            dist.all_gather(all_label, label.contiguous())

            all_example_index = [torch.zeros_like(example_index) for _ in range(self.args.num_gpus)]
            dist.all_gather(all_example_index, example_index.contiguous())

            all_candidate_items = [torch.zeros_like(candidate_items) for _ in range(self.args.num_gpus)]
            dist.all_gather(all_candidate_items, candidate_items.contiguous())

            all_real_label = [torch.zeros_like(real_label) for _ in range(self.args.num_gpus)]
            dist.all_gather(all_real_label, real_label.contiguous())

            predict_score, label = self.clean_dist_duplicate(all_predict_score, all_label, all_example_index)
            candidate_items, _ = self.clean_dist_duplicate(all_candidate_items, all_label, all_example_index)
            real_label, _ = self.clean_dist_duplicate(all_real_label, all_label, all_example_index)

        recall = cal_recall(label.cpu(), predict_score.cpu(), [10, 50, 100, 200])
        ndcg = cal_ndcg(label.cpu(), predict_score.cpu(), [10, 50, 100, 200])
        gini = cal_gini(predict_score.cpu(), self.args.item_count, [10, 50, 100, 200])
        cratio = cal_cratio(predict_score.cpu(), self.args.item2pop, [10, 50, 100, 200])

        if self.args.gpu == 0:
            info(f"\nRecall:{recall}\nNDCG:{ndcg}\nGini:{gini}\nCRatio:{cratio}")

        flag_metrics = ndcg[-1]
        if flag_metrics > self.best_valid_result:
            self.early_stop_step = 0
            self.best_valid_result = flag_metrics
        else:
            self.early_stop_step += 1

        if self.early_stop_step > 10:
            save = False
            exit_ = True
        elif self.early_stop_step > 0:
            save = False
            exit_ = False
        else:
            save = True
            exit_ = False
        if self.args.train_stage == 2:
            save = True
            exit_ = False

        return {'save': save, 'exit': exit_, 'result': [flag_metrics]}

    def clean_dist_duplicate(self, all_predict_score, all_label, all_example_index):
        all_predict_score = torch.concat(all_predict_score, dim=0).cpu()
        all_label = torch.concat(all_label, dim=0).cpu()

        predict_score = torch.zeros_like(all_predict_score)
        label = torch.zeros_like(all_label)
        example_index = torch.concat(all_example_index, dim=0).cpu()

        predict_score[example_index] = all_predict_score
        label[example_index] = all_label
        exp_cnt = max(example_index) + 1

        return predict_score[:exp_cnt], label[:exp_cnt]


    def transfer_device(self, data):
        device = next(self.model.parameters()).device
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)

    def save(self, path):
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        saved_parameters = {}
        model_generator = self.model.named_parameters() if not self.args.distributed else self.model.module.named_parameters()
        for param_name, param in model_generator:
            if param.requires_grad:
                saved_parameters[param_name] = param
        torch.save(saved_parameters, path)

    def load(self, path, loc=None):
        weights = torch.load(path, map_location=next(self.model.parameters()).device)
        if self.args.distributed:
            info(self.model.module.load_state_dict(weights, strict=False))
        else:
            info(self.model.load_state_dict(weights, strict=False))

    def save_pickle(self, obj, path):
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        pickle.dump(obj, open(path, 'wb'))

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        model_generator = self.model.named_parameters() if not self.args.distributed else self.model.module.named_parameters()
        for _, param in model_generator:
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )




def main_worker(gpu, args):
    args.gpu = gpu
    args.rank = gpu
    info(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        args.world_size = args.num_gpus
        args.dist_backend = "nccl"
        args.dist_url = f'tcp://127.0.0.1:{args.port}'
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    info(f'Building train loader at GPU {gpu}')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.root_path + args.backbone)
    train_loader, valid_loader, test_loader = get_dataloader(args, tokenizer)
    trainer = Trainer(args, tokenizer, train_loader, valid_loader, test_loader, train=True)
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    args.dataset = args.dataset
    info("============runner run with args=================")
    info(args)
    gpu_count = args.num_gpus
    if args.distributed:
        gpu_count = torch.cuda.device_count()
        mp.spawn(main_worker, (args,), nprocs=gpu_count, join=True)
    else:
        main_worker(0, args)
