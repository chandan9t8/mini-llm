import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_in, dim_out, context_length, dropout=0.1):
        """
        dim_in - dim of the input token
        dim_out - dim of the output token(combined dimension of all heads)
        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.dim_per_head = dim_out // num_heads

        #weight matrics for query, key, value
        self.W_q = nn.Linear(dim_in, dim_out, bias=False)
        self.W_k = nn.Linear(dim_in, dim_out, bias=False)
        self.W_v = nn.Linear(dim_in, dim_out, bias=False)

        #dropout layer
        self.dropout = nn.Dropout(dropout)

        #causal mask to ensure that each token can only attend to previous tokens
        #a triangular matrix with 0s in lower triangle and 1s in upper triangle
        self.register_buffer('mask',
                             torch.triu(torch.ones(context_length, context_length),
                                        diagonal=1)
                             )
        #output linear projection
        self.out_proj = nn.Linear(dim_out, dim_out)

    def forward(self, x):
        batch, num_tokens, dim_in = x.shape

        #calc Q, K, V matrices 
        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)

        #split complete Q, K, V into multiple heads(split the dim of each token into num_heads and dim per head)
        keys = keys.view(batch, num_tokens, self.num_heads, self.dim_per_head)
        queries = queries.view(batch, num_tokens, self.num_heads, self.dim_per_head)
        values = values.view(batch, num_tokens, self.num_heads, self.dim_per_head)

        #transpose to get num_heads in the 2nd dimension(helps in parallel computation of all heads)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        queries = queries.transpose(1,2)

        #attention scores
        attn_scores = queries @ keys.transpose(2,3)

        #slice the big mask to current seq length
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)         #replaces "True" in upper triangle with -inf

        # attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # apply dropout mask
        attn_weights = self.dropout(attn_weights)

        # context vector
        context_vector = attn_weights @ values
        context_vector = context_vector.transpose(1,2)   # convert back to (b, num_tokens, num_heads, dim_per_head)

        context_vector = context_vector.contiguous().view(batch, num_tokens, self.dim_out)

        # linear projection : glean together info from all heads
        context_vector = self.out_proj(context_vector)

        return context_vector
    
class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))   #scale and shift are learnable parameters

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * x_norm + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.layer_norm1 = LayerNorm(cfg["emb_dim"])
        self.layer_norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.mha = MultiHeadAttention(cfg["n_heads"], cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"], cfg["dropout"])
        self.ffn = FeedForward(cfg["emb_dim"])

    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut

        return x