import os
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np

def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx

def get_raw_dataset_split_index(output_path,
                                dataset_name,
                                seed,
                                split_name,
                                data_size,
                                rebuild=False):

    shuffle_idx = get_shuffle_idx(seed, data_size)
    return shuffle_idx[0:data_size].tolist()

class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        batch = {}
        pad_token_id = data[-1][-1]

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        ### make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(prompt,
                                    pad=(0, pad_length),
                                    mode='constant',
                                    value=pad_token_id)
            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(0, pad_length),
                                             mode='constant',
                                             value=0)
        else:
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        batch["prompt"] = batch["prompt"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch

class PromptDataset(Dataset):

    def __init__(self, prompt_dataset,
                 pad_token_id) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.pad_token_id = pad_token_id

    def __len__(self):
        length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id

# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        if os.path.exists(dataset_name):
            self.raw_datasets = load_from_disk(dataset_name)
        elif not dataset_name == 'local/jsonfile':
            self.raw_datasets = load_dataset(dataset_name)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return

def create_dataset_split(current_dataset, raw_dataset, tokenizer, max_seq_len):
    prompt_dataset = []
    filtered = 0
    for i, tmp_data in enumerate(current_dataset):
        # tokenize the text
        prompt = raw_dataset.get_prompt(tmp_data)
        if prompt is not None:
            prompt_token = tokenizer(prompt, return_tensors="pt")
            if prompt_token["input_ids"].size()[-1] <= max_seq_len:
                for key_word in ["input_ids", "attention_mask"]:
                    prompt_token[key_word] = prompt_token[
                        key_word].squeeze(0).flip(0)
                prompt_dataset.append(prompt_token)
            else:
                filtered += 1
    print(f'Creating dataset {raw_dataset.dataset_name_clean} '
            f'size={len(prompt_dataset)} {filtered=}')
    return PromptDataset(prompt_dataset, tokenizer.pad_token_id)

def create_dataset(local_rank, output_path,
                   seed, tokenizer, max_seq_len):
    raw_dataset = DahoasFullhhrlhfDataset(output_path=output_path, 
                                            seed=seed, 
                                            local_rank=local_rank, 
                                            dataset_name="Dahoas/full-hh-rlhf"
                                            )
    train_dataset = raw_dataset.get_train_data()
    train_index = get_raw_dataset_split_index(output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, 
                                              "train",
                                              len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, 
                                            raw_dataset,
                                            tokenizer,
                                            max_seq_len)
    return train_dataset

# English dataset
class DahoasFullhhrlhfDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/full-hh-rlhf"
        self.dataset_name_clean = "Dahoas_full_hh_rlhf"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']