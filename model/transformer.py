import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads=8):
        """Implementation of multi-head attention layer.

        Args:
            embed_size (int): token's dimension, i.e. word embedding vector size
            heads (int, optional): number of distinct representations to learn. Defaults to 8.
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        
        self.head_dim, mod = divmod(embed_size, heads)
        assert mod == 0, "Embedding size needs to be fully divided by heads"
        
        # self.to_qvk = nn.Linear(self.head_dim, self.head_dim * 3, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        
    def forward(self, values, keys, queries, mask):
    # def forward(self, x, mask): 
        # assert x.dim() == 3, "3D tensor must be provided"
        
        # qkv = self.to_qvk(x) # [batch, token, dim*3*heads]
        
        # queries, keys, values = tuple([rearrange(qkv, 'b t (d k h) -> k b h t d', k=3, h=self.heads)]) # [3, batch, heads, tokens, heads_dim]
        
        # energy = torch.einsum('b h i d , b h j d -> b h i j', queries, keys) / (self.head_dim ** (1/2)) # [batch, heads, tokens(query_len), tokens(key_len)]
        
        batch = queries.shape[0] # how many examples are sended in at the same time
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        # split embedding into self.heads pieces
        values = values.reshape(batch, value_len, self.heads, self.head_dim) # (batch, value_len, heads, heads_dim)
        keys = keys.reshape(batch, key_len, self.heads, self.head_dim) # (batch, key_len, heads, heads_dim)
        queries = queries.reshape(batch, query_len, self.heads, self.head_dim) # (batch, query_len, heads, heads_dim)
        
        values = self.values(values)
        keys = self.values(keys)
        queries = self.values(queries)
        
        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys]) # (batch, heads, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -np.inf)
        
        attention = torch.softmax(energy, dim=3)
        
        out = torch.einsum("b h i j, b h j d -> b h i d", attention, values) # [batch, token, heads, heads_dim]
        out = rearrange(out, "b h t d -> b t (h d)") # [batch, token, heads*heads_dim]
        out = self.fc_out(out)
        
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        """Vanilla transformer block

        Args:
            embed_size (int): token's vector length
            heads (int): number of heads
            dropout (float): probability of dropping values
            forward_expansion (int): intermediate number of nodes
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(forward_expansion*embed_size, embed_size), 
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, values, keys, queries, mask):
        attention = self.attention(values, keys, queries, mask)
        out = self.norm1(self.dropout(attention) + queries)
        out = self.norm2(self.feed_forward(out) + out)
        
        return out
    

class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        embed_size, 
        blocks, 
        heads, 
        forward_expansion, 
        dropout, 
        max_length,
        device
    ):
        super(TransformerEncoder, self).__init__()
        # self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.block_list = [TransformerBlock(embed_size, heads, dropout, forward_expansion) for block in range(blocks)]
        self.layers = nn.ModuleList(self.block_list)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, values, keys, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(values, keys, query, src_mask)
        
        return out

class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        trg_vocab_size, 
        embed_size, 
        blocks, 
        heads, 
        forward_expansion, 
        dropout, 
        max_length, 
        device
    ):
        super(TransformerDecoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.block_list = [DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for block in range(blocks)]
        self.layers = nn.ModuleList(self.block_list)
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            out = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        out = self.fc_out(out)
        
        return out

class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        trg_vocab_size, 
        src_pad_index, 
        trg_pad_index, 
        embed_size=256, 
        blocks=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        max_length=100, 
        device="cuda"
    ):
        super(Transformer, self).__init__()
        
        self.encoder = TransformerEncoder(src_vocab_size, embed_size, blocks, heads, forward_expansion, dropout, max_length, device)
        self.decoder = TransformerDecoder(trg_vocab_size, embed_size, blocks, heads, forward_expansion, dropout, max_length, device)
        
        self.src_pad_index = src_pad_index
        self.trg_pad_index = trg_pad_index
        self.device = device
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_index).unsqueeze(1).unsqueeze(2) # (batch, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        batch, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(batch, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    
    x = torch.tensor([[1,3,5,6,7,8,9,7], [1,5,2,3,6,4,3,7]]).to(device)
    trg = torch.tensor([[1,3,4,5,7,8,7], [1,3,4,3,6,9,7]]).to(device)
    
    src_pad_idx=0
    trg_pad_idx=0
    src_vocab_size=10
    trg_vocab_size=10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape)