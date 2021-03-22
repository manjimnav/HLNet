import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
        
        
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, has_minute=False, has_hour=True):
        super(TemporalEmbedding, self).__init__()

        hour_size = 25
        weekday_size = 7
        day_size = 32
        month_size = 13
        minute_size = 6

        Embed = nn.Embedding
        self.has_minute=has_minute
        self.has_hour=has_hour
        
        if has_minute:
            self.minute_embed = Embed(minute_size, d_model)
        if has_hour:
            self.hour_embed = Embed(hour_size, d_model)
            
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        if self.has_minute:
            minute_x = self.minute_embed(x[:, :, 4])
        if self.has_hour:
            hour_x = self.hour_embed(x[:, :, 3])
            
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        
        result = weekday_x + day_x + month_x
        if self.has_hour:
            result += hour_x
            
        if self.has_minute:
            result += minute_x
            
        return  result


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, has_minute=False, has_hour=True):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, has_minute=has_minute, has_hour=has_hour)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
    
        x = self.value_embedding(x)
        if x_mark is not None:
            x += self.temporal_embedding(x_mark)

        return self.dropout(x)
        
class InformerDataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, has_minute=False, has_hour=True):
        super(InformerDataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, has_minute=has_minute, has_hour=has_hour)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x) + self.position_embedding(x)

        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark)
        
        return self.dropout(x)
