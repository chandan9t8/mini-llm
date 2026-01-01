import torch
import torch.nn as nn
from src.modules import TransformerBlock, LayerNorm

class MiniLLM(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.token_embedding_layer = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_embedding_layer = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.dropout = nn.Dropout(cfg["dropout"])

        self.transformer_blocks = nn.Sequential(*(TransformerBlock(cfg) for _ in range(cfg["n_layers"])))

        self.final_norm = LayerNorm(cfg["emb_dim"])

        self.linear_output_layer = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape

        token_embeddings = self.token_embedding_layer(token_ids)
        positional_embeddings = self.pos_embedding_layer(torch.arange(sequence_length, device=token_ids.device))
        x = token_embeddings + positional_embeddings

        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.linear_output_layer(x)
        return logits