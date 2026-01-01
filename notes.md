# Data preparation

`data.py`
- load raw data in memory
- divide the raw text into train & val
- create batches of inputs and targets
    -  use `tiktoken` library to get encoder and decoder of `gpt2` and encode the entire text(split into individual words & convert to tokenIDs)
    - iterate the seq and create input target pairs
    - func `create_dataloader()` that uses `Dataloader` class to load the batches of inputs and targets
- create separate batches for train & val sets.
- convert tokenIDs to embedding vectors.

`modules.py`
- mha module
- layer norm


# Parameters and what they mean

`context_length`
- property of the model, defines the hard limit of the model
- defines how far the model can look
- it controls the size of the positional embedding matrix of the model

`max_length`
- property of data preprocessing pipeline
- defines the way we chunk/chop the data

`max_length` <= `context_length`
usually both are set equal to reduce the complexity of the code.


# Dimensions

**data preparation**
token embedding layer : `vocab_size` x `emb_dim` 
positional embedding layer : `context_length` x `emb_dim`

**multihead attention module**
input - (`batch_size`, `max_length`, `emb_dim`)
output - (`batch_size`, `max_length`, `emb_dim`)

- weight matrices Q, K, V : `dim_in` x `dim_out`
    - `dim_in` --> `dims of the input token(token's embedding size)` 
    - `dim_out` --> `dims of output we want`
    - usually `dim_out = each head dim x num_heads`
- Q, K, V matrices

**Layer Norm**
input - (`batch_size`, `max_length`, `emb_dim`)
output - (`batch_size`, `max_length`, `emb_dim`)

**feed forward network**
input - (`batch_size`, `max_length`, `emb_dim`)
output - (`batch_size`, `max_length`, `emb_dim`)

**Transformer block**
layer norm -> MHA -> layer norm -> ffn
input - (`batch_size`, `max_length`, `emb_dim`)
output - (`batch_size`, `max_length`, `emb_dim`)
 
  
**final linear layer**
input - (`batch_size`, `max_length`, `emb_dim`)
output - (`batch_size`, `emb_dim`, `vocab_size`)

**Complete LLM**
input - (`batch_size`, `max_size(or context_length)`)
output - (`batch_size`, `context_length`, `vocab_size`)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

  
  

# Architecture

llm-from-scratch/
├── configs/                # Configuration files (YAML)
│   ├── pretrain_config.yaml
│   └── finetune_config.yaml
├── data/                   # Raw and processed data (GitIgnore this!)
│   ├── raw/
│   └── processed/
├── notebooks/              # For experimentation/exploration only
│   ├── 01_data_exploration.ipynb
│   └── 02_attention_mechanism_test.ipynb
├── outputs/                # Model checkpoints and logs
│   ├── logs/
│   └── models/
├── src/                    # The actual source code package
│   └── minigpt/            # Name your package (e.g., minigpt)
│       ├── __init__.py
│       ├── config.py       # Pydantic models for config validation
│       ├── data.py         # Tokenizers and Dataset classes
│       ├── model.py        # The GPT Architecture (or broken down)
│       ├── modules.py      # Building blocks (Attention, FeedForward)
│       ├── train.py        # Training loops
│       ├── generate.py     # Inference/Text generation logic
│       └── utils.py        # Helper functions (plotting, seeding)
├── tests/                  # Unit tests
│   ├── test_attention.py
│   └── test_dataloader.py
├── .gitignore
├── Makefile                # Shortcuts for commands
├── pyproject.toml          # Dependencies (Modern replacement for requirements.txt)
└── README.md
