import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model import IAM4VP

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CrossAttention(nn.Module):
    def __init__(self, video_dim, tensor_dim):
        super(CrossAttention, self).__init__()

        self.video_dim = video_dim
        self.tensor_dim = tensor_dim

        self.video_key = nn.Linear(self.video_dim, self.tensor_dim)
        self.video_value = nn.Linear(self.video_dim, self.tensor_dim)

        self.tensor_query = nn.Linear(self.tensor_dim, self.tensor_dim)

    def forward(self, video_tensor, other_tensor):
        B, T, C1, H, W = video_tensor.shape
        other_tensor = other_tensor.permute(1, 0, 2)

        video_tensor_flattened = video_tensor.view(B, T, -1)

        video_K = self.video_key(video_tensor_flattened)
        video_V = self.video_value(video_tensor_flattened)

        tensor_Q = self.tensor_query(other_tensor)

        attention_scores = torch.bmm(tensor_Q, video_K.transpose(1, 2))
        attention_probs = F.softmax(attention_scores, dim=-1)

        attention_output = torch.bmm(attention_probs, video_V)

        return attention_output

class Encoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads):
        super(Encoder, self).__init__()
        self.embed = nn.Linear(2, embed_size)  # Convert (x, y) to a specified embedding sizei
        self.pos = PositionalEncoding(embed_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, heads),
            num_layers
        )

    def forward(self, src):
        src = self.embed(src)
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch, features)
        src = self.pos(src)
        return self.transformer(src)

class Decoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, output_seq_len):
        super(Decoder, self).__init__()
        self.embed = nn.Linear(2, embed_size)
        self.pos = PositionalEncoding(embed_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_size, heads),
            num_layers
        )
        self.fc_out = nn.Linear(embed_size, 2)
        self.output_seq_len = output_seq_len

    def forward(self, enc_out):
        # Using the last position of the encoder's output as the initial input for the decoder
        trg = enc_out[-1].unsqueeze(0).repeat(self.output_seq_len, 1, 1)
        trg = self.pos(trg)
        out = self.transformer(trg, enc_out)
        return self.fc_out(out.permute(1, 0, 2))


class TrajectoryTransformer(nn.Module):
    def __init__(self, input_seq_len, output_seq_len, video_dim, tensor_dim, embed_size=64, num_layers=3, heads=4):
        super(TrajectoryTransformer, self).__init__()

        self.encoder = Encoder(embed_size, num_layers, heads)
        self.iam4vp = IAM4VP([12,9,240,320])
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.cross_attention = CrossAttention(video_dim, tensor_dim)
        self.decoder = Decoder(embed_size, num_layers, heads, output_seq_len)

    def forward(self, src, video_tensor):
        enc_out = self.encoder(src)
        video_tensor1, video_tensor2 = self.iam4vp(video_tensor)
        video_tensor1 = self.avg(video_tensor1)
        cross_att_out = self.cross_attention(video_tensor1, enc_out)
        cross_att_out = cross_att_out.permute(1, 0, 2)
        return self.decoder(cross_att_out)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_seq_len = 8
    output_seq_len = 12
    video_dim = 64  # Embedding size to match flattened video tensor dimensions
    tensor_dim = 64  # Assuming encoder output dimension is 64
    embed_size = 64
    num_layers = 3
    heads = 4

    model = TrajectoryTransformer(input_seq_len, output_seq_len, video_dim, tensor_dim, embed_size, num_layers, heads).to(device)

    src = torch.randn((4, 8, 2)).to(device)  # Example input batch of size 256
    video_tensor = torch.randn((4, 12, 9, 240, 320)).to(device)  # Example video tensor
    out = model(src, video_tensor)
    print(out.shape)
