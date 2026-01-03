import torch
import logging

from src.data import download_data, read_data, split_data, create_dataloader
from src.config_loader import config
from src.model import MiniLLM
from src.generate import generate_text


logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

URL = ("https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch02/01_main-chapter-code/"
"the-verdict.txt")
FILE_PATH = "data/the-verdict.txt"

MAX_LENGTH = config['max_length']
STRIDE = config['stride']
BATCH_SIZE = config['batch_size']

VOCAB_SIZE = config['vocab_size']
EMB_DIM = config['emb_dim']
CONTEXT_LENGTH = config['context_length']
NUM_HEADS = config['n_heads']
DROPOUT = config['dropout']

def main():
    logger.info("================== Welcome to mini LLM ==================")
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


    logger.info("==================== Data preparation pipeline completed successfully. ====================")

if __name__ == "__main__":
    main()