# Tokenizers and Dataset classes

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import requests

# download data from url
def download_data(url, file_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # raise error for HTTP issues
        with open(file_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False


# read downloaded data
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

# divide into train and val
def split_data(raw_text, split_ratio=0.9):
    split_idx = int(len(raw_text) * split_ratio)
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]
    return train_data, val_data

# data class for creating datasets
class CreateDataset(Dataset):
    
    def __init__(self, raw_text, tokenizer, max_length, stride):
        self.input_tokens = []
        self.target_tokens = []

        token_ids = tokenizer.encode(raw_text)

        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_tokens.append(torch.tensor(input_chunk))
            self.target_tokens.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_tokens)

    def __getitem__(self, idx):
        return self.input_tokens[idx], self.target_tokens[idx]

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")

    # create instance of the data class
    dataset = CreateDataset(txt, tokenizer, max_length, stride)

    # create the dataloader -- it a wrapper around the dataset
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader


