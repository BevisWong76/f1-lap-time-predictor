import random
import torch
from torch import nn  

class F1LapSeq2Seq(nn.Module):

    def __init__(
        self,
        cat_dims,
        cont_dim,
        hidden_dim=128,
        embed_dim=8,
        forecast_len=5,
        num_layers=2,
        dropout=0.3,
        target_cont_idx=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.forecast_len = forecast_len
        self.target_cont_idx = target_cont_idx

        # Categorical Embedding layers
        self.embeddings = nn.ModuleList(
            [nn.Embedding(d, embed_dim) for d in cat_dims]
        )
        self.total_cat_dim = len(cat_dims) * embed_dim
        self.input_dim = self.total_cat_dim + cont_dim

        # Encoder & Decoder LSTMs
        self.encoder = nn.LSTM(
            self.input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.decoder = nn.LSTM(
            self.input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def _embed_and_concat(self, x_cat, x_cont):
        """Helper to process categorical embeddings and combine with continuous features in one pass."""
        cat_embs = [
            emb(x_cat[:, :, j]) for j, emb in enumerate(self.embeddings)
        ]
        cat_emb = torch.cat(cat_embs, dim=-1)
        return torch.cat([cat_emb, x_cont], dim=-1)

    def forward(
        self,
        x_cat,
        x_cont,
        x_cat_fut,
        x_cont_fut,
        y_target=None,
        teacher_forcing_ratio=0.5,
        noise_std=0.0,
    ):
        # 1. ENCODER PASS
        x_enc = self._embed_and_concat(x_cat, x_cont)
        _, (hidden, cell) = self.encoder(x_enc)

        # 2. AUTOREGRESSIVE DECODER PASS
        predictions = []
        curr_hidden = (hidden, cell)

        # Seed step t=0 with the last known historical target value (t=-1)
        last_pred = x_cont[:, -1, self.target_cont_idx]

        for t in range(self.forecast_len):
            curr_cont = x_cont_fut[:, t : t + 1, :].clone()

            # Teacher forcing decision
            use_teacher = (
                self.training
                and (y_target is not None)
                and (random.random() < teacher_forcing_ratio)
            )

            if use_teacher:
                if t == 0:
                    # Target at t=0 is the last known historical observation
                    target_val = x_cont[:, -1, self.target_cont_idx]
                else:
                    # Target at step t is ground truth of the PREVIOUS future step (t-1)
                    target_val = y_target[:, t - 1, 0]

                if noise_std > 0.0:
                    target_val = (
                        target_val
                        + torch.randn_like(target_val) * noise_std
                    )

                curr_cont[:, 0, self.target_cont_idx] = target_val
            else:
                # Use model's previous prediction
                curr_cont[:, 0, self.target_cont_idx] = last_pred.view(-1)

            curr_cat = x_cat_fut[:, t : t + 1, :]
            x_dec_step = self._embed_and_concat(curr_cat, curr_cont)

            dec_out, curr_hidden = self.decoder(x_dec_step, curr_hidden)
            step_pred = self.fc(
                self.dropout(dec_out)
            )  # Shape: [batch_size, 1, 1]

            predictions.append(step_pred)

            # Always detach predictions fed into subsequent autoregressive inputs
            last_pred = step_pred.squeeze(1).squeeze(-1).detach()

        return torch.cat(predictions, dim=1)