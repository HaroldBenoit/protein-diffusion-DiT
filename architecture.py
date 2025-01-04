
### Code inspired by https://github.com/facebookresearch/DiT

import torch.nn as nn
import torch 
import math
import einops

import dataclasses


@dataclasses.dataclass
class ArchitectureConfig:

    ## input/output
    input_dim:int = 12
    learn_sigma:bool = True ## whether we learn the variance of the noise

    ### model params
    d_model: int = 128
    debug: bool = True
    layer_norm_eps: float = 1e-6
    init_range: float = 0.02
    seq_len: int = 256

    ## attention
    d_head: int = 64
    n_heads: int = 2
    n_layers: int = 8
    qkv_bias:bool = True
    attn_proj_bias:bool = True
    dropout:float = 0.0

    ## mlp
    mlp_ratio: int = 4
    mlp_bias: bool = True

    ## embedding
    frequency_embedding_size:int = 256

    learned_pos_embed: bool = False


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Positional                #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, config, ):
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(config.frequency_embedding_size, config.d_model, bias=True),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.config.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

## from https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py


## fixed
def SinusoidalPosEncoding(config):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if config.d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(config.d_head_model))
    pe = torch.zeros(config.seq_len, config.d_model)
    position = torch.arange(0, config.seq_len).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, config.d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / config.d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


#################################################################################
#                                 Core DiT Model                                #
#################################################################################



class Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.qkv_bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.attn_proj_bias)

        self.resid_dropout = nn.Dropout(config.dropout)

        self.config = config
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            raise ValueError("FA requires PyTorch >= 2.0")

    def forward(self, x, attn_mask=None):
      # x: (batch seq_len d_model)

        q, k ,v  = self.qkv(x).split(self.config.d_model, dim=2)

        q = einops.rearrange(q, "batch seq_len (n_heads d_head) -> batch n_heads seq_len d_head", n_heads=self.config.n_heads, d_head=self.config.d_head)
        k = einops.rearrange(k, "batch seq_len (n_heads d_head) -> batch n_heads seq_len d_head", n_heads=self.config.n_heads, d_head=self.config.d_head)
        v = einops.rearrange(v, "batch seq_len (n_heads d_head) -> batch n_heads seq_len d_head", n_heads=self.config.n_heads, d_head=self.config.d_head)



        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout, is_causal=False) # (batch n_heads seq_len d_head)

        y = einops.rearrange(y, "batch n_heads seq_len d_head -> batch seq_len (n_heads d_head)") # (batch seq_len d_model)

        # output projection
        y = self.resid_dropout(self.proj(y))

        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.d_model, config.mlp_ratio * config.d_model, bias=config.mlp_bias)
        self.c_proj = nn.Linear(config.mlp_ratio * config.d_model, config.d_model, bias=config.mlp_bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.d_model, elementwise_affine=False, eps=config.layer_norm_eps)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.d_model, elementwise_affine=False, eps=config.layer_norm_eps)

        self.mlp = MLP(config)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.d_model, 6 * config.d_model, bias=True)
        )

    def forward(self, x, timestep_embedding, attn_mask=None) :
        ## get shift, scale and gating based on timestep embedding
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(timestep_embedding).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm_final = nn.LayerNorm(config.d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.d_model, 2 * config.d_model, bias=True)
        )

        self.mean_linear = nn.Linear(config.d_model, config.input_dim, bias=True)
        if self.config.learn_sigma:
            self.sigma_linear = nn.Linear(config.d_model, config.input_dim, bias=True)

    def forward(self, x, timestep_embedding):
        shift, scale = self.adaLN_modulation(timestep_embedding).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        if not self.config.learn_sigma:
            return self.mean_linear(x)
        else:
            return self.mean_linear(x), self.sigma_linear(x)



class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.config = config

        self.x_embedder = nn.Linear(config.input_dim, config.d_model)
        self.t_embedder = TimestepEmbedder(config)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, config.seq_len, config.d_model), requires_grad=config.learned_pos_embed)


        self.blocks = nn.ModuleList([
            DiTBlock(config) for _ in range(config.n_layers)
        ])
        self.final_layer = FinalLayer(config)
        self.initialize_weights()

    @staticmethod
    def model_name(config):
        s=f"DiT-n_layers={config.n_layers}-d_model={config.d_model}-d_heads={config.n_heads}-learn_sigma={config.learn_sigma}"

        if config.learned_pos_embed:
            s+="-learned_pos_embed=True"

        return s

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        if not self.config.learned_pos_embed:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            pos_embed = SinusoidalPosEncoding(self.config)
            self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))

        else:
            torch.nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)


        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        ## Init such that the full DiT block is the identity function, helps with training supposedly

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)


        # Zero-out output layers for identity function:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.mean_linear.weight, 0)
        nn.init.constant_(self.final_layer.mean_linear.bias, 0)


    def forward(self, x, t, attn_mask=None, eps=1e-3):
        """
        Forward pass of DiT.
        x: (batch_size, seq_len, input_dim)
        t: (batch_size,) tensor of diffusion timesteps
        """
        x_in = x

        x = self.x_embedder(x) + self.pos_embed  # (batch_size, seq_len, d_model)
        t = self.t_embedder(t)                   # (batch_size, d_model)
        for block in self.blocks:
            x = block(x, t, attn_mask)                      # (batch_size, seq_len, d_model)
        x = self.final_layer(x, t)                # (batch_size, seq_len, output_dim)


        if self.config.learn_sigma:
            a, b = x
            b = b.sigmoid() * (1 - eps * 2) + eps

            return (x_in - a) / b

        else:
            a = x
            return x_in - a
        
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
