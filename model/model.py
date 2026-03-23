from typing import Tuple, List, Optional

import torch
import torch.nn as nn

'''
Inspiration:
https://openaccess.thecvf.com/content/ACCV2022W/TCV/papers/Sridhar_Transformer_Based_Motion_In-Betweening_ACCVW_2022_paper.pdf
https://github.com/Pavi114/motion-completion-using-transformers
'''

class InputEncoder(nn.Module):
    def __init__(
        self,
        num_joints,
        joint_embedding_size,
        root_embedding_size
    ):
        super(InputEncoder, self).__init__()

        self.J = num_joints
        self.joint_embedding_size = joint_embedding_size
        self.root_embedding_size = root_embedding_size

        self.rot_encoder = nn.Sequential(
            nn.Linear(in_features=6, out_features=16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=self.joint_embedding_size)
        )
        self.pos_encoder = nn.Sequential(
            nn.Linear(in_features=3, out_features=8),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=8, out_features=self.root_embedding_size)
        )

    def forward(
        self,
        local_6d_rot: torch.Tensor,
        global_root_pos: torch.Tensor
    ) -> torch.Tensor:
        B, T, J, _ = local_6d_rot.shape
        assert J == self.J

        local_6d_rot = self.rot_encoder(local_6d_rot)
        global_root_pos = self.pos_encoder(global_root_pos)

        # Reshape local_6d_rot
        local_6d_rot = local_6d_rot.reshape((B, T, J * self.joint_embedding_size))
        
        seq = torch.cat([global_root_pos, local_6d_rot], dim=-1)
        return seq
#InputEncoder


class OutputDecoder(nn.Module):
    def __init__(
        self,
        num_joints,
        joint_embedding_size,
        root_embedding_size
    ):
        super(OutputDecoder, self).__init__()

        self.J = num_joints
        self.joint_embedding_size = joint_embedding_size
        self.root_embedding_size = root_embedding_size

        self.rot_decoder = nn.Sequential(
            nn.Linear(in_features=self.joint_embedding_size, out_features=16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=6)
        )
        self.pos_decoder = nn.Sequential(
            nn.Linear(in_features=self.root_embedding_size, out_features=8),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=8, out_features=3)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        global_root_pos = x[:, :, :self.root_embedding_size]
        local_6d_rot = x[:, :, self.root_embedding_size:]

        B, T, _ = local_6d_rot.shape

        # Reshape local_6d_rot back
        local_6d_rot = local_6d_rot.reshape(B, T, self.J, self.joint_embedding_size)

        local_6d_rot = self.rot_decoder(local_6d_rot)
        global_root_pos = self.pos_decoder(global_root_pos)
        return local_6d_rot, global_root_pos


class SinusoidalPositionalEncoding(nn.Module):
    # https://medium.com/@lixue421/understanding-positional-encoding-in-transformers-2c7336728be5
    def __init__(
        self,
        dim: int,
        max_len: int = 5000
    ):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # shape (max_len, dim)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if self.dim != x.size(-1):
            raise ValueError(f'Positional encoding dim ({self.dim}) != input last dim ({x.size(-1)})')

        pe = self.pe[:T].unsqueeze(0).to(x.dtype).to(x.device)  # (1, T, D)
        return x + pe
#SinusoidalPositionalEncoding


class MotionTransformer(nn.Module):
    def __init__(
        self,
        num_joints: int,
        joint_embedding_size: int,
        root_embedding_size: int,

        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        dropout: float,

        max_len: int
    ):
        super().__init__()

        self.num_joints = num_joints
        self.joint_embedding_size = joint_embedding_size
        self.root_embedding_size = root_embedding_size

        self.input_encoder = InputEncoder(
            num_joints=num_joints,
            joint_embedding_size=joint_embedding_size,
            root_embedding_size=root_embedding_size
        )

        self.output_decoder = OutputDecoder(
            num_joints=num_joints,
            joint_embedding_size=joint_embedding_size,
            root_embedding_size=root_embedding_size
        )

        self.dim_model = num_joints * joint_embedding_size + root_embedding_size
        assert self.dim_model % num_heads == 0, "dim_model must be divisible by num_heads"

        # self.transformer = nn.Transformer(
        #     d_model=self.dim_model,
        #     nhead=num_heads,
        #     num_encoder_layers=num_encoder_layers,
        #     num_decoder_layers=num_decoder_layers,
        #     dropout=dropout,
        #     batch_first=True
        # )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.pos_encoder = SinusoidalPositionalEncoding(
            dim=self.dim_model,
            max_len=max_len
        )

        # self.mask_token = nn.Parameter(torch.randn(1, 1, self.dim_model) * 0.02)
    
    # OLD VERSION OF TRANSFORMER TRAINING - MASKED FRAME PREDICTION, encoder-decoder structure
    # def forward(
    #     self,
    #     src_rot: torch.Tensor,
    #     src_pos: torch.Tensor,
    #     # tgt_rot: Optional[torch.Tensor] = None,
    #     # tgt_pos: Optional[torch.Tensor] = None,
    #     fixed_points: List[int] = []
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     # assert (tgt_rot is None and tgt_pos is None) or type(tgt_rot) == torch.Tensor and type(tgt_pos) == torch.Tensor, "tgt_rot and tgt_pos must be both None or both Tensors"

    #     # Encoding
    #     enc_seq = self.input_encoder(src_rot, src_pos)
    #     B, T, _ = enc_seq.shape

    #     enc_h = self.pos_encoder(enc_seq)
    #     dec_h = enc_h.clone()

    #     # mask_bool = torch.ones(T, dtype=torch.bool, device=enc_seq.device)
    #     # mask_bool[fixed_points] = False     # True = masked, False = available
    #     # mask_token_exp = self.mask_token.expand(B, T, self.dim_model)
    #     # dec_seq = torch.where(mask_bool.view(1, T, 1), mask_token_exp.to(dec_seq.dtype), dec_seq)

    #     # dec_h = self.pos_encoder(dec_seq)

    #     tgt_mask = torch.full((T, T), float("-inf"), device=enc_h.device, dtype=enc_h.dtype)
    #     for j in fixed_points:
    #         tgt_mask[:, j] = 0.0
    #     tgt_mask = tgt_mask.to(enc_h.device).to(enc_h.dtype)

    #     out = self.transformer(enc_h, dec_h, tgt_mask=tgt_mask)

    #     pred_rot, pred_pos = self.output_decoder(out)
    #     return pred_rot, pred_pos

    def forward(
        self,
        src_rot: torch.Tensor,
        src_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoding
        enc_seq = self.input_encoder(src_rot, src_pos)
        # B, T, _ = enc_seq.shape
        enc_h = self.pos_encoder(enc_seq)

        out = self.transformer(enc_h)

        pred_rot_delta, pred_pos_delta = self.output_decoder(out)

        # Residual connection
        pred_rot = src_rot + pred_rot_delta
        pred_pos = src_pos + pred_pos_delta
        # pred_rot, pred_pos = pred_rot_delta, pred_pos_delta

        return pred_rot, pred_pos
    
#MotionTransformer
