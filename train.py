import argparse

import torch
import time
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama import LlamaTokenizer

import data_utils

def parse_args():
    parser = argparse.ArgumentParser(
        description="(Step 3) RLHF training arguments")

    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/root/.cache',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        default='openlm-research/open_llama_3b_easylm',
        help=
        "Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--critic_model_name_or_path",
        type=str,
         default='openlm-research/open_llama_3b_easylm',
        help=
        "Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--per_device_generation_batch_size",
        type=int,
        default=1,
        help=
        "Batch size (per device) for the training dataloader and generation purpose."
    )
    parser.add_argument(
        "--per_device_training_batch_size",
        type=int,
        default=1,
        help=
        "Mini Batch size (per device) for the training dataloader and training purpose."
    )
    parser.add_argument("--max_prompt_seq_len",
        type=int,
        default=256,
        help="The maximum sequence length.")
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help=
        "Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature."
    )
    args = parser.parse_args()
    return args

def train():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    args = parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(
            'openlm-research/open_llama_3b_easylm', 
            fast_tokenizer=True, 
            legacy=True
    )

    data_collator = data_utils.DataCollatorRLHF(args.max_prompt_seq_len,
                                     args.inference_tp_size)

    prompt_train_dataset = data_utils.create_dataset(local_rank=rank,
                                                     output_path=args.data_output_path,
                                                     seed = 114514,
                                                     tokenizer=tokenizer,
                                                     max_seq_len=args.max_prompt_seq_len,
                                                     )

    print(prompt_train_dataset)
    
    prompt_train_sampler = DistributedSampler(prompt_train_dataset)
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_generation_batch_size)

    print(f"process {rank} has created dataloader")

    model = AutoModelForCausalLM.from_pretrained(
        'openlm-research/open_llama_3b_easylm',
        trust_remote_code=True,
        tokenizer=tokenizer
    ).to(f"cuda:{rank}")

    print(f"process {rank} has initialized model")

    time.sleep(100)
    exit(0)

    # optimizer specific arguments e.g. LR, momentum, etc...
    base_optimizer_arguments = { "lr": 1e-4}

    # Wrap a base optimizer into OSS
    base_optimizer = torch.optim.Adam  # any pytorch compliant optimizer
    optimizer = OSS(
        params=model.parameters(),
        optim=base_optimizer,
        **base_optimizer_arguments)

    # Wrap the model into ShardedDDP, which will reduce gradients to the proper ranks
    model = ShardedDDP(model, optimizer)

    # Any relevant training loop, nothing specific to OSS. For example:
    model.train()
    for e in range(epochs):
        for data in prompt_train_dataloader:
            data = data.to(rank)
            # Train
            model.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train()