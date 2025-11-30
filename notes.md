# Data preparation

`data.py`
- load raw data in memory
- divide the raw text into train & val
- create batches of inputs and targets
    - use `tiktoken` library to get encoder and decoder of `gpt2` and encode the entire text(split into individual words & convert to tokenIDs)
    - iterate the seq and create input target pairs
    - func `create_dataloader()` that uses `Dataloader` class to load the batches of inputs and targets
- create separate batches for train & val sets.









# Parameters and what they mean
`context_length`
- property of the model
- defines the hard limit of the model
- it controls the size of the positional embedding matrix of the model

`max_length`
- property of data preprocessing pipeline
- defines the way we chunk/chop the data

`max_length` <= `context_length`
usually both are set equal to reduce the complexity of the code.
