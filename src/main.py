import torch
import logging

from data import *
from config_loader import config
from modules import MultiHeadAttention


logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

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
NUM_HEADS = config['model']['n_heads']
DROPOUT = config['model']['dropout']

def main():
    logger.info("Starting the mini LLM...")
    logger.info("================== Starting data preparation pipeline ==================")
    try:
        # download data
        logger.info("Downloading data...")
        download_flag = download_data(URL, FILE_PATH)
        if download_flag:
            logger.info("Download successful...")
        else:
            print("Data download failed.")
    except Exception as e:
        logger.error(f"Error occurred during data download: {e}")

    # read data
    try:
        logger.info("Reading data...")
        raw_text = read_data(FILE_PATH)
        logger.info(f"The len of raw data is {len(raw_text)}")
    except Exception as e:
        logger.error(f"An error occurred during data reading: {e}")

    # divide raw text into train and val
    try:
        logger.info("Splitting data into train and val...")
        train_data, val_data = split_data(raw_text, split_ratio=0.8)
        logger.info(f"Len of train set: {len(train_data)} , val set: {len(val_data)}")

    except Exception as e:
        logger.error(f"An error occurred during data splitting: {e}")

    # create dataloader
    torch.manual_seed(123)

    logger.info(f"creating dataloaders with batch_size={BATCH_SIZE}, max_length={MAX_LENGTH}, stride={STRIDE}")
    train_loader = create_dataloader(train_data, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, stride=STRIDE, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, stride=STRIDE, shuffle=False)
    logger.info(f"The len of train dataloader: {len(train_loader)}, val loader: {len(val_loader)}")
    for input, target in train_loader:
        logger.info(f"Input batch shape: {input.shape}, Target batch shape: {target.shape}")
        break

    # convert tokensIDs to embeddings
    token_embedding_layer = torch.nn.Embedding(VOCAB_SIZE, EMB_DIM)
    pos_embedding_layer = torch.nn.Embedding(CONTEXT_LENGTH, EMB_DIM)

    train_input_embeddings_list = []
    for inputs, _ in train_loader:
        token_embeddings = token_embedding_layer(inputs)
        seq_length = inputs.size(1)
        pos_indices = torch.arange(seq_length).unsqueeze(0)
        positional_embeddings = pos_embedding_layer(pos_indices)
        train_input_embeddings_list.append(token_embeddings + positional_embeddings)
    train_input_embeddings = torch.cat(train_input_embeddings_list, dim=0)
    logger.info(f"Train input embeddings shape: {train_input_embeddings.shape}")
    
    val_input_embeddings_list = []
    for inputs, _ in val_loader:
        token_embeddings = token_embedding_layer(inputs)
        seq_length = inputs.size(1)
        pos_indices = torch.arange(seq_length).unsqueeze(0)
        positional_embeddings = pos_embedding_layer(pos_indices)
        val_input_embeddings_list.append(token_embeddings + positional_embeddings)
    val_input_embeddings = torch.cat(val_input_embeddings_list, dim=0)
    logger.info(f"Val input embeddings shape: {val_input_embeddings.shape}")

    logger.info("******************* Data preparation pipeline completed successfully. *******************")


if __name__ == "__main__":
    main()