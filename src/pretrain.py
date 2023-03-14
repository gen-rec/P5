import json
import pickle
import os.path
from pathlib import Path
from typing import Optional

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, T5TokenizerFast

from dist_utils import reduce_dict
from param import parse_args
from pretrain_data import get_loader
from utils import LossMeter

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase


# The Trainer inherits TrainerBase in trainer_base.py
class Trainer(TrainerBase):
    def __init__(
            self, args, train_loader=None, val_loader=None, test_loader=None, train=True, tokenizer=None,
            extra_token_embedding: Optional[torch.Tensor] = None
    ):
        """
        Trainer for P5 model

        :param args: Namespace object containing all the arguments
        :param train_loader: Train dataloader
        :param val_loader: Validation dataloader
        :param test_loader: Test dataloader
        :param train: Whether to train or not
        :param tokenizer: Tokenizer object to use
        :param extra_token_embedding: If provided, use this as extra token embedding(# of tokens, embedding dim)
        """
        super().__init__(
                args,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                train=train
        )

        assert args.whole_word_embed
        from pretrain_model import P5Pretraining

        model_kwargs = {}
        model_class = P5Pretraining

        config = self.create_config()
        assert tokenizer is not None, "Tokenizer must be provided."
        self.tokenizer: T5TokenizerFast = tokenizer
        self.model = self.create_model(model_class, config, **model_kwargs)

        ####
        # Initialize new token embeddings
        prev_vocab_size = 32_100
        print(f"** {prev_vocab_size} tokens in original tokenizer **")
        print(f"** Resizing token embeddings from {self.model.shared.weight.shape[0]} to {len(self.tokenizer.vocab)}")

        new_token_embedding = self.model.resize_token_embeddings(len(self.tokenizer.vocab))

        if extra_token_embedding is None:
            # Random initialize
            nte_mean = torch.mean(new_token_embedding.weight[:prev_vocab_size]).item()
            nte_std = torch.std(new_token_embedding.weight[:prev_vocab_size]).item()

            print(f"Randomly initializing new tokens... (mean: {nte_mean:.3f}, std: {nte_std:.3f})")

            torch.nn.init.normal_(new_token_embedding.weight[prev_vocab_size:], mean=nte_mean, std=nte_std)
        else:
            # Load extra token embedding
            print("Loading from extra token embedding...")
            assert extra_token_embedding.shape[0] == len(self.tokenizer.vocab) - prev_vocab_size, \
                f"Extra token embedding size {extra_token_embedding.shape[0]} does not match " \
                f"the number of new tokens {len(self.tokenizer.vocab) - prev_vocab_size}"
            assert extra_token_embedding.shape[1] == self.model.shared.weight.shape[1], \
                f"Extra token embedding dimension {extra_token_embedding.shape[1]} does not match " \
                f"the model dimension {self.model.shared.weight.shape[1]}"

            with torch.no_grad():
                new_token_embedding.weight[prev_vocab_size:] = extra_token_embedding

        print(
                f"** {len(self.tokenizer.vocab) - prev_vocab_size} new tokens initialized "
                f"({new_token_embedding.weight[prev_vocab_size:].shape}) **"
        )
        ####

        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)
            self.start_epoch = int(args.load.split('Epoch-')[-1])

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                        self.model, self.optim, opt_level='O1', verbosity=self.verbose
                )

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu], find_unused_parameters=True)
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME

        if self.args.dry:
            results = self.evaluate_epoch(epoch=0)

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_eval_loss = 100000.

            if 't5' in self.args.backbone:
                project_name = "P5_Pretrain"

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        for epoch in range(self.args.epoch):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            # Train
            self.model.train()

            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=240)

            epoch_results = {}
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):

                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

                if self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()

                # self.model.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 200:
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()

            dist.barrier()

            results = reduce_dict(epoch_results, average=False)
            if self.verbose:
                train_loss = results['total_loss']
                train_loss_count = results['total_loss_count']

                avg_train_loss = train_loss / train_loss_count
                losses_str = f"Train Loss: {avg_train_loss:.3f}\n"

                for name, loss in results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(results[name + '_count'])
                        if loss_count > 0:
                            avg_loss = loss / loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                losses_str += '\n'
                print(losses_str)

            dist.barrier()

            if args.run_valid:
                # Validation
                valid_results = self.evaluate_epoch(epoch=epoch)

                valid_results = reduce_dict(valid_results, average=False)
                if self.verbose and step_i % 200:
                    valid_loss = valid_results['total_loss']
                    valid_loss_count = valid_results['total_loss_count']

                    avg_valid_loss = valid_loss / valid_loss_count
                    losses_str = f"Valid Loss: {avg_valid_loss:.3f}\n"

                    for name, loss in valid_results.items():
                        if name[-4:] == 'loss':
                            loss_count = int(valid_results[name + '_count'])
                            if loss_count > 0:
                                avg_loss = loss / loss_count
                                losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                    losses_str += '\n'
                    print(losses_str)

                dist.barrier()

                if self.verbose:
                    # Save
                    if avg_valid_loss < best_eval_loss:
                        best_eval_loss = avg_valid_loss
                        self.save("BEST_EVAL_LOSS")
                    self.save("Epoch%02d" % (epoch + 1))

                dist.barrier()
            else:
                # Skip validation
                print("Skip validation for Epoch%02d" % (epoch + 1))
                self.save("Epoch%02d" % (epoch + 1))

                dist.barrier()

    def evaluate_epoch(self, epoch):
        LOSSES_NAME = self.args.LOSSES_NAME

        epoch_results = {}
        for loss_name in LOSSES_NAME:
            epoch_results[loss_name] = 0.
            epoch_results[f'{loss_name}_count'] = 0

        self.model.eval()
        with torch.no_grad():
            if self.verbose:
                loss_meter = LossMeter()
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

                pbar = tqdm(total=len(self.val_loader), ncols=240)

            for step_i, batch in enumerate(self.val_loader):

                if self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.valid_step(batch)

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 200:
                    desc_str = f'Valid Epoch {epoch} |'
                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)
                dist.barrier()

            if self.verbose:
                pbar.close()
            dist.barrier()

            return epoch_results


def main_worker(gpu, args):
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')
    # define the prompts used in training
    if args.train == 'yelp':
        train_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
                           'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11',
                                          '2-12'],
                           'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9'],
                           'review': ['4-1', '4-2'],
                           'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
                           }
    else:
        train_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
                           'sequential': ['2-1', '2-7', '2-11'],
                           'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10',
                                           '3-11'],
                           'review': ['4-1', '4-2', '4-3'],
                           'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
                           }
    if args.task_index is not None:
        print(f"Task index: {args.task_index}")
        train_task_list = list(train_task_list.items())
        train_task_list = dict((train_task_list[args.task_index],))
        print(f"Train task list: {train_task_list}")

    # define sampling numbers for each group of personalized prompts (see pretrain_data.py)
    # if greater than 1, a data sample will be used for multiple times with different prompts in certain task family
    train_sample_numbers = {'rating': 1, 'sequential': (5, 5, 10), 'explanation': 1, 'review': 1,
                            'traditional': (10, 5)}

    ####
    tokenizer: PreTrainedTokenizerFast = T5TokenizerFast.from_pretrained("t5-small", model_max_length=512)

    datamaps = json.load(open(os.path.join("data", args.train, "datamaps.json"), "r", encoding="utf-8"))
    user_ids = [f"user_{u}" for u in datamaps["user2id"].values()]
    item_ids = [f"item_{i}" for i in datamaps["item2id"].values()]
    user_item_ids = user_ids + item_ids

    num_added_tokens = tokenizer.add_tokens(user_item_ids, special_tokens=True)
    print(f"Added {num_added_tokens} tokens to the tokenizer", flush=True)

    os.makedirs(os.path.join(args.output, f"tokenizer-{args.rank}"), exist_ok=True)
    tokenizer.save_pretrained(os.path.join(args.output, f"tokenizer-{args.rank}"))

    # Load token embedding
    if args.extra_token_embedding is None:
        embedding_weight = None
    else:
        extra_token_embedding = pickle.load(open(args.extra_token_embedding, "rb"))
        embedding_weight = extra_token_embedding["token_embedding"]
    ####

    train_loader = get_loader(
            args,
            train_task_list,
            train_sample_numbers,
            split=args.train,
            mode='train',
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed,
            tokenizer=tokenizer
    )

    print(f'Building val loader at GPU {gpu}')
    # define the prompts used in validation
    if args.valid == 'yelp':
        val_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
                         'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11',
                                        '2-12'],
                         'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9'],
                         'review': ['4-1', '4-2'],
                         'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
                         }
    else:
        val_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
                         'sequential': ['2-1', '2-7', '2-11'],
                         'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11'],
                         'review': ['4-1', '4-2', '4-3'],
                         'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
                         }
    if args.task_index is not None:
        print(f"Task index: {args.task_index}")
        val_task_list = list(val_task_list.items())
        val_task_list = dict((val_task_list[args.task_index],))

    val_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
    val_loader = get_loader(
            args,
            val_task_list,
            val_sample_numbers,
            split=args.valid,
            mode='val',
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed,
            tokenizer=tokenizer
    )

    args.verbose = True

    trainer = Trainer(args, train_loader, val_loader, train=True, tokenizer=tokenizer,
                      extra_token_embedding=embedding_weight)
    trainer.train()


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    if args.local_rank in [0, -1]:
        print(args)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
    if args.local_rank in [0, -1]:
        print(LOSSES_NAME)
    LOSSES_NAME.append('total_loss')

    args.LOSSES_NAME = LOSSES_NAME

    comments = []
    dsets = []
    if 'toys' in args.train:
        dsets.append('toys')
    if 'beauty' in args.train:
        dsets.append('beauty')
    if 'sports' in args.train:
        dsets.append('sports')
    if 'yelp' in args.train:
        dsets.append('yelp')
    comments.append(''.join(dsets))
    if args.backbone:
        comments.append(args.backbone)
    comments.append(''.join(args.losses.split(',')))
    if args.comment != '':
        comments.append(args.comment)
    comment = '_'.join(comments)

    from datetime import datetime

    current_time = datetime.now().strftime('%b%d_%H-%M')

    project_dir = Path(__file__).resolve().parent.parent

    if args.local_rank in [0, -1]:
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'
        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)

