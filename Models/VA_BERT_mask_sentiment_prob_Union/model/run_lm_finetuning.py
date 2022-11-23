# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import json
import pandas as pd
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)

from BertForMaskVA import BertForMaskVA

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from IPython import embed

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskVA, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}
                         

class TextDataset(Dataset):
    def __init__(self, tokenizer, sentiment_dict_V, sentiment_dict_A, args, file_path="train", sentiment_w_path_V='sentiment_V', sentiment_w_path_A='sentiment_A', add_tokens_data='tokens', block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_name_or_path + "_cached_lm_" + str(block_size)  + "_fold" + args.fold + "_" + filename
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            #with open(file_path, encoding="utf-8") as f:
                #text = f.read()

            df = pd.read_csv(file_path,header=None)
            text = [([w.split('(')[:-1][0] for w in i.split(' ')]) for i in df[1]]
            label_A = df[3]
            label_V = df[4]
            for i in range(len(text)):
                token = tokenizer.encode(text[i],pad_to_max_length=True,max_length=len(max(text, key=len)))
                #prob
                token_V = [sentiment_dict_V[i] if (i in sentiment_dict_V) else 0 for i in token]
                token_A = [sentiment_dict_A[i] if (i in sentiment_dict_A) else 0 for i in token]
                token.extend(token_V)
                token.extend(token_A)
                token.append(label_A[i].astype('float32'))
                token.append(label_V[i].astype('float32'))
                self.examples.append(token)
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def load_and_cache_examples(args, tokenizer, sentiment_dict_V, sentiment_dict_A,evaluate=False, test=False):
    dataset = TextDataset(
        tokenizer,
        sentiment_dict_V, 
        sentiment_dict_A,
        args,
        file_path=args.eval_data_file if evaluate else args.test_data_file if test else args.train_data_file,
        sentiment_w_path_V = args.sentiment_w_path_V,
        sentiment_w_path_A = args.sentiment_w_path_A,
        add_tokens_data = args.add_tokens_data,
        block_size=args.block_size,
    )
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, sentiment_mask_V, sentiment_mask_A, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    if args.distribution == 'bernoulli':
        DIST = torch.bernoulli
    #sentiment word mask's probability
        
    labels = inputs.clone()
    labels2 = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix=sentiment_mask_V.float()*args.mlm_probability
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    #special_tokens_mask=1 -> probability=0
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = DIST(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced_V = DIST(torch.full(labels.shape, 1)).bool() & masked_indices


    probability_matrix=sentiment_mask_A.float()*args.mlm_probability
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels2.tolist()
    ]
    #embed()
    #special_tokens_mask=1 -> probability=0
    try:
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    except:
        embed()
    masked_indices = DIST(probability_matrix).bool()
    labels2[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced_A = DIST(torch.full(labels.shape, 0.8)).bool() & masked_indices
    indices_replaced = indices_replaced_V | indices_replaced_A
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    embed()
    # 10% of the time, we replace masked input tokens with random word
    #indices_random = DIST(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    #random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    #inputs[indices_random] = random_words[indices_random]
   
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def eval(V, A, valence, arousal):
    #accuracy = (pred == label).sum().item()/label.size(0)
    mae_V = (V - valence).abs().mean().item()
    mae_A = (A - arousal).abs().mean().item()
    rmse_V = (V - valence).pow(2).mean().sqrt().item()
    rmse_A = (A - arousal).pow(2).mean().sqrt().item()
    r_V = np.corrcoef(np.array(V.cpu()).ravel(), np.array(valence.cpu()).ravel())[0,1]
    r_A = np.corrcoef(np.array(A.cpu()).ravel(), np.array(arousal.cpu()).ravel())[0,1]
    return mae_V, mae_A, rmse_V, rmse_A, r_V, r_A

def train(args, train_dataset, model, tokenizer, sentiment_dict_V, sentiment_dict_A):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
        # tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    best_val_loss = float('inf')
    loss_df = pd.DataFrame({'epoch':[], 'loss':[]})
    i=0
    for _ in train_iterator:
        i+=1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            label_V = batch[:,-2:-1]
            label_A = batch[:,-1:]
            batch = batch[:,:-2]

            sentiment_A = batch[:, int(2*len(batch[0])/3):]
            sentiment_V = batch[:, int(len(batch[0])/3): int(2*len(batch[0])/3)]
            batch = batch[:,:int(len(batch[0])/3)].long()

            inputs, labels = mask_tokens(batch, tokenizer, sentiment_V, sentiment_A, args) if args.mlm else (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            label_V = label_V.to(args.device)
            label_A = label_A.to(args.device)
            model.train()
            outputs = model(inputs, masked_lm_labels=labels, label_V=label_V, label_A=label_A, alpha=args.alpha) if args.mlm else model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #     # Log metrics
                #     if (
                #         args.local_rank == -1 and args.evaluate_during_training
                #     ):  # Only evaluate when single GPU otherwise metrics may not average well
                #         results = evaluate(args, model, tokenizer)
                #         # for key, value in results.items():
                #             # tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                #     # tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                #     # tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                #     logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    print('start evaluate')
                    results = evaluate(args, model, tokenizer, sentiment_dict_V, sentiment_dict_A,eval_type='val')
                    checkpoint_prefix = "checkpoint"
                    print('best loss', best_val_loss)
                    print('loss now', results['best_loss'])
                    loss_df = loss_df.append({'epoch':i, 'loss':results['best_loss']}, ignore_index=True)
                    loss_df.to_csv('./'+args.output_dir+'/val_loss_df.csv')
                    if best_val_loss > results['best_loss']:
                        best_val_loss = results['best_loss']
                        print('save best model')
                        results = evaluate(args, model, tokenizer, sentiment_dict_V, sentiment_dict_A,eval_type='val', save=True)
                        results = evaluate(args, model, tokenizer, sentiment_dict_V, sentiment_dict_A,eval_type='test', save=True)
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))#少一個global_step
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    else:
                        print('dont save')
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # if args.local_rank in [-1, 0]:
        # tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, sentiment_dict_V, sentiment_dict_A, prefix="", eval_type='val', save=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if eval_type == 'val':
        eval_dataset = load_and_cache_examples(args, tokenizer, sentiment_dict_V, sentiment_dict_A, evaluate=True)
    else:
        eval_dataset = load_and_cache_examples(args, tokenizer, sentiment_dict_V, sentiment_dict_A, test=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    best_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    all_predict_V = []
    all_predict_A = []
    all_label_V = []
    all_label_A = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
    #for batch in eval_dataloader:
        label_V = batch[:,-2:-1]
        label_A = batch[:,-1:]
        batch = batch[:,:-2]
        sentiment_A = batch[:, int(2*len(batch[0])/3):].long()
        sentiment_V = batch[:, int(len(batch[0])/3): int(2*len(batch[0])/3)].long()
        batch = batch[:, :int(len(batch[0])/3)].long()

        inputs, labels = mask_tokens(batch, tokenizer, sentiment_V, sentiment_A, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        label_V = label_V.to(args.device)
        label_A = label_A.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels, label_V=label_V, label_A=label_A, alpha=args.alpha) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            VA_loss = outputs[1]
            eval_loss += lm_loss.mean().item()
            best_loss += VA_loss.mean().item()
        nb_eval_steps += 1
        all_label_V += label_V.cpu().tolist()
        all_label_A += label_A.cpu().tolist()
        all_predict_V += outputs[2].cpu().tolist()
        all_predict_A += outputs[3].cpu().tolist()

    result = pd.DataFrame({'index':range(len(all_label_V)),
                           'labelV':all_label_V,
                           'labelA':all_label_A,
                           'predictV':all_predict_V,
                           'predictA':all_predict_A})
    if save==True:
        result.to_csv(args.output_dir+'/{}_predict{}.csv'.format(eval_type, str(args.mlm_probability)))

        mae_V, mae_A, rmse_V, rmse_A, r_V, r_A = eval(torch.tensor(all_predict_V), torch.tensor(all_predict_A), torch.tensor(all_label_V), torch.tensor(all_label_A))
        dict_={}
        dict_['mae_V'] = mae_V
        dict_['mae_A'] = mae_A
        dict_['rmse_V'] = rmse_V
        dict_['rmse_A'] = rmse_A
        dict_['r_V'] = r_V
        dict_['r_A'] = r_A

        with open(args.output_dir+'/{}_evaluation_result{}.json'.format(eval_type, str(args.mlm_probability)), 'w') as fp:
            json.dump(dict_, fp)

    eval_loss = eval_loss / nb_eval_steps
    best_loss = best_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity,
              "eval_loss": eval_loss,
              "best_loss": best_loss} #加一個loss

    output_eval_file = os.path.join(eval_output_dir, prefix, "{}_eval_results.txt".format(eval_type))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default="../data/train0.csv", type=str, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default="../data/val0.csv",
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )

    parser.add_argument(
        "--test_data_file",
        default="../data/test.csv",
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--fold",
        default="0",
        type=str,
        help="which fold is trainning right now",
    )
    parser.add_argument("--model_type", default="bert", type=str, help="The model architecture to be fine-tuned.")
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-cased",
        type=str,
        help="The model checkpoint for weights initialization.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--sentiment_w_path_V",
        type=str,
        default='./wikitext-2/mask_prob_V.json',
        help='sentiment word txt file_path',
    )
    parser.add_argument(
        "--sentiment_w_path_A",
        type=str,
        default='./wikitext-2/mask_prob_A.json',
        help='sentiment word txt file_path',
    )
    parser.add_argument(
        "--add_tokens_data",
        type=str,
        default='./wikitext-2/DVA_dataset.csv',
        help='add tokens csv file_path',
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default='bernoulli',
        help='distribution name',
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="loss weighted sum of VA_loss and mlm_loss, alpha means the weight of VA_loss"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.block_size <= 0:
        args.block_size = (
            tokenizer.max_len_single_sentence
        )  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    df = pd.read_csv(args.add_tokens_data,header=None)
    text = [([w.split('(')[:-1][0] for w in i.split(' ')]) for i in df[1]]
    word = set()
    for i in df[1]:
        word.update([w.split('(')[:-1][0] for w in i.split(' ')])
    tokenizer.add_tokens(list(word))

    f = open(args.sentiment_w_path_V, 'r')
    sentiment_dict_V = json.load(f)
    f.close()

    f = open(args.sentiment_w_path_A, 'r')
    sentiment_dict_A = json.load(f)
    f.close()

    d2 = sentiment_dict_V.copy()
    for i in tqdm(d2.keys()):
        if len(i.strip())!=1:
            nt = tokenizer.encode(i.strip())#[1]
            sentiment_dict_V[nt[1]] = sentiment_dict_V[i]
            sentiment_dict_A[nt[1]] = sentiment_dict_A[i]
            del sentiment_dict_V[i]
            del sentiment_dict_A[i]
        else:
            del sentiment_dict_V[i]
            del sentiment_dict_A[i]

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, sentiment_dict_V, sentiment_dict_A, evaluate=False)
        model.resize_token_embeddings(len(tokenizer))
        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, sentiment_dict_V, sentiment_dict_A)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, sentiment_dict_V, sentiment_dict_A, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
