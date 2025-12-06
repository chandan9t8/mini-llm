from .data import *
from config_loader import config

URL = ("https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch02/01_main-chapter-code/"
"the-verdict.txt")
FILE_PATH = "data/the-verdict.txt"

MAX_LENGTH = config['dataset']['max_length']
STRIDE = config['dataset']['stride']
BATCH_SIZE = config['dataset']['batch_size']

VOCAB_SIZE = config['model']['vocab_size']
EMB_DIM = config['model']['emb_dim']
CONTEXT_LENGTH = config['model']['context_length']

def main():
    print("************   Welcome to Mini-LLM!   *************")

    # download data
    download_flag = download_data(URL, FILE_PATH)
    if download_flag:
        print("Data downloaded successfully.")
    else:
        print("Data download failed.")

    # read data
    raw_text = read_data(FILE_PATH)
    print(f"The len of raw data is {len(raw_text)}")

    # divide raw text into train and val
    train_data, val_data = split_data(raw_text, split_ratio=0.8)
    print(f"The len of train data is {len(train_data)}")
    print(f"The len of val data is {len(val_data)}")

    # create dataloader
    torch.manual_seed(123)

    train_loader = create_dataloader(train_data, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, stride=STRIDE, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, stride=STRIDE, shuffle=False)
    print(f"The len of train dataloader is {len(train_loader)}")
    print(f"The len of val dataloader is {len(val_loader)}")

    # convert tokensIDs to embeddings
    token_embedding_layer = torch.nn.Embedding(VOCAB_SIZE, EMB_DIM)
    pos_embedding_layer = torch.nn.Embedding(CONTEXT_LENGTH, EMB_DIM)

    for inputs, _ in train_loader:
        token_embeddings = get_tokens_embeddings(inputs, token_embedding_layer)
        seq_length = inputs.size(1)
        positional_embeddings = get_positional_embeddings(seq_length, pos_embedding_layer)
        positional_embeddings = positional_embeddings.unsqueeze(0)
        train_input_embeddings = token_embeddings + positional_embeddings
    
    for inputs, _ in val_loader:
        token_embeddings = get_tokens_embeddings(inputs, token_embedding_layer)
        seq_length = inputs.size(1)
        positional_embeddings = get_positional_embeddings(seq_length, pos_embedding_layer)
        positional_embeddings = positional_embeddings.unsqueeze(0)
        val_input_embeddings = token_embeddings + positional_embeddings


if __name__ == "__main__":
    main()