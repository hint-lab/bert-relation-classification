import argparse
import glob
import logging
import os
import sys
import random
import torch.nn as nn
import numpy as np
import torch
import socket
# wss
# import ptvsd
# Allow other computers to attach to ptvsd at this IP address and port.
# ptvsd.enable_attach(address=('192.168.11.2', 3000), redirect_output=True)
# Pause the program until a remote debugger is attached
# ptvsd.wait_for_attach()


from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
# from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils import (RELATION_LABELS, compute_metrics, convert_examples_to_features,
                   output_modes, data_processors)
import torch.nn.functional as F

from argparse import ArgumentParser
from config import Config
from model import BertForSequenceClassification
logger = logging.getLogger(__name__)
#additional_special_tokens = ["[E11]", "[E12]", "[E21]", "[E22]"]
additional_special_tokens = []
#additional_special_tokens = ["e11", "e12", "e21", "e22"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config, train_dataset, model, tokenizer):
    """ Train the model """
    config.train_batch_size = config.per_gpu_train_batch_size * \
        max(1, config.n_gpu)
    if config.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=config.train_batch_size)

    if config.max_steps > 0:
        t_total = config.max_steps
        config.num_train_epochs = config.max_steps // (
            len(train_dataloader) // config.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=config.warmup_steps, t_total=t_total)
    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if config.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank],
                                                          output_device=config.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                config.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                config.train_batch_size * config.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if config.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(config.num_train_epochs),
                            desc="Epoch", disable=config.local_rank not in [-1, 0])
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(config.seed)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=config.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(config.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      # XLM and RoBERTa don't use segment_ids
                      'token_type_ids': batch[2],
                      'labels':      batch[3],
                      'e1_mask': batch[4],
                      'e2_mask': batch[5],
                      }

            outputs = model(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]
            if config.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % config.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if config.local_rank in [-1, 0] and config.logging_steps > 0 and global_step % config.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if config.local_rank == -1 and config.evaluate_during_training:
                        results = evaluate(config, model, tokenizer)
                    logging_loss = tr_loss
                # if config.local_rank in [-1, 0] and config.save_steps > 0 and global_step % config.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(
                #         config.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     # Take care of distributed/parallel training
                #     model_to_save = model.module if hasattr(
                #         model, 'module') else model
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(config, os.path.join(
                #         output_dir, 'training_config.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)

            if config.max_steps > 0 and global_step > config.max_steps:
                epoch_iterator.close()
                break
        if config.max_steps > 0 and global_step > config.max_steps:
            train_iterator.close()
            break
    return global_step, tr_loss / global_step


def evaluate(config, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task = config.task_name
    eval_output_dir = config.output_dir

    results = {}
    eval_dataset = load_and_cache_examples(
        config, eval_task, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and config.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    config.eval_batch_size = config.per_gpu_eval_batch_size * \
        max(1, config.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        eval_dataset) if config.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=config.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", config.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(config.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      # XLM and RoBERTa don't use segment_ids
                      'token_type_ids': batch[2],
                      'labels':      batch[3],
                      'e1_mask': batch[4],
                      'e2_mask': batch[5],
                      }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(eval_task, preds, out_label_ids)
    results.update(result)
    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    output_eval_file = "eval/sem_res.txt"
    with open(output_eval_file, "w") as writer:
        for key in range(len(preds)):
            writer.write("%d\t%s\n" %
                         (key+8001, str(RELATION_LABELS[preds[key]])))
    return result


def load_and_cache_examples(config, task, tokenizer, evaluate=False):
    if config.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    processor = data_processors[config.task_name]()
    # Load data features from cache or dataset file
    # cached_features_file = os.path.join(config.data_dir, 'cached_{}_{}_{}_{}'.format(
    #     'dev' if evaluate else 'train',
    #     list(filter(None, 'bert-large-uncased'.split('/'))).pop(),
    #     str(config.max_seq_len),
    #     str(task)))
    # if os.path.exists(cached_features_file):
    #     logger.info("Loading features from cached file %s",
    #                 cached_features_file)
    #     features = torch.load(cached_features_file)
    # else:
    logger.info("Creating features from dataset file at %s",
                config.data_dir)
    label_list = processor.get_labels()
    examples = processor.get_dev_examples(
        config.data_dir) if evaluate else processor.get_train_examples(config.data_dir)
    features = convert_examples_to_features(
        examples, label_list, config.max_seq_len, tokenizer, "classification", use_entity_indicator=config.use_entity_indicator)
    # if config.local_rank in [-1, 0]:
    #     logger.info("Saving features into cached file %s",
    #                 cached_features_file)
    #     torch.save(features, cached_features_file)

    if config.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    output_mode = "classification"
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_e1_mask = torch.tensor(
        [f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor(
        [f.e2_mask for f in features], dtype=torch.long)  # add e2 mask
    if output_mode == "classification":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_input_mask,
                            all_segment_ids, all_label_ids, all_e1_mask, all_e2_mask)
    return dataset


def main():
    parser = ArgumentParser(
        description="BERT for relation extraction (classification)")
    parser.add_argument('--config', dest='config')
    args = parser.parse_args()
    config = Config(args.config)

    if os.path.exists(config.output_dir) and os.listdir(config.output_dir) and config.train and not config.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(config.output_dir))

    # Setup CUDA, GPU & distributed training
    if config.local_rank == -1 or config.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
        config.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(config.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        config.n_gpu = 1
    config.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if config.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   config.local_rank, device, config.n_gpu, bool(config.local_rank != -1))

    # Set seed
    set_seed(config.seed)

    # Prepare GLUE task
    processor = data_processors["semeval"]()
    output_mode = output_modes["semeval"]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if config.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    # Make sure only the first process in distributed training will download model & vocab
    bertconfig = BertConfig.from_pretrained(
        config.pretrained_model_name, num_labels=num_labels, finetuning_task=config.task_name)
    # './large-uncased-model', num_labels=num_labels, finetuning_task=config.task_name)
    bertconfig.l2_reg_lambda = config.l2_reg_lambda
    bertconfig.latent_entity_typing = config.latent_entity_typing
    if config.l2_reg_lambda > 0:
        logger.info("using L2 regularization with lambda  %.5f",
                    config.l2_reg_lambda)
    if config.latent_entity_typing:
        logger.info("adding the component of latent entity typing: %s",
                    str(config.latent_entity_typing))
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True, additional_special_tokens=additional_special_tokens)
    # 'bert-large-uncased', do_lower_case=True, additional_special_tokens=additional_special_tokens)
    model = BertForSequenceClassification.from_pretrained(
        config.pretrained_model_name, config=bertconfig)
    # './large-uncased-model', config=bertconfig)

    if config.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(config.device)

    # logger.info("Training/evaluation parameters %s", config)

    # Training
    if config.train:
        train_dataset = load_and_cache_examples(
            config, config.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(config, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s",
                    global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if config.train and (config.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(config.output_dir) and config.local_rank in [-1, 0]:
            os.makedirs(config.output_dir)

        logger.info("Saving model checkpoint to %s", config.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(config, os.path.join(
            config.output_dir, 'training_config.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(
            config.output_dir)
        tokenizer = BertTokenizer.from_pretrained(
            config.output_dir, do_lower_case=True, additional_special_tokens=additional_special_tokens)
        model.to(config.device)

    # Evaluation
    results = {}
    if config.eval and config.local_rank in [-1, 0]:
        tokenizer = BertTokenizer.from_pretrained(
            config.output_dir, do_lower_case=True, additional_special_tokens=additional_special_tokens)
        checkpoints = [config.output_dir]
        if config.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(
                glob.glob(config.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                '-')[-1] if len(checkpoints) > 1 else ""
            model = BertForSequenceClassification.from_pretrained(checkpoint)
            model.to(config.device)
            result = evaluate(config, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v)
                          for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
