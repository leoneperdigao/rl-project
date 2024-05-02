import torch
import torch.nn.functional as F
import math

class Attention(torch.nn.Module):
    """Multi-head scaled dot-product ego-oppo-attention.

    This class implements an attention mechanism where 'ego' and 'oppo' inputs are processed
    differently, reflecting a domain-specific requirement. The attention mechanism used here
    is the scaled dot-product attention.

    Parameters:
    ego_dim (int): Dimension of ego input.
    oppo_dim (int): Dimension of opponent input.
    embed_dim (int, optional): Total dimension of the model's output and inner state. Default: 96.
    num_heads (int, optional): Number of attention heads. Default: 3.
    """
    def __init__(self, ego_dim, oppo_dim, embed_dim=96, num_heads=3):
        super(Attention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear transformations for query, key, and value projections
        self.q_proj = torch.nn.Linear(ego_dim, embed_dim)
        self.kv_proj = torch.nn.Linear(oppo_dim, 2*embed_dim)
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim)

        self.__set_parameters()
        
    def forward(self, ego, opponent):
        """Forward pass of the multi-head attention mechanism.

        Parameters:
        ego (Tensor): The ego tensor.
        opponent (Tensor): The opponent tensor.

        Returns:
        Tensor: The output tensor after applying attention, adjusted to input batch format.
        """
        # Check if input is batched, adjust if necessary
        is_batched = ego.dim() > 1
        if not is_batched:
            ego = ego.unsqueeze(0)
            opponent = opponent.unsqueeze(0)
        batch_size = ego.size(0)

        # Compute queries, keys, and values with appropriate reshaping and permutation
        q = self.q_proj(ego).view(batch_size, self.num_heads, 1, self.head_dim)
        kv = self.kv_proj(opponent).view(batch_size, opponent.size(1), self.num_heads, 2*self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)

        # Calculate attention scores using scaled dot-product
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits /= math.sqrt(self.head_dim)  # Scale by sqrt of dimensionality of key vectors
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v).view(batch_size, self.embed_dim)

        # Final output projection
        o = self.o_proj(values)

        return o if is_batched else o.squeeze(0)
    
    def __set_parameters(self):
        """Initializes the parameters of the module using Xavier uniform initialization for weights and
        setting biases to zero. This type of initialization helps in keeping the gradient magnitudes
        manageable, preventing vanishing or exploding gradients early in training."""
        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
