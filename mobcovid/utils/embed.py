import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        a = position * div_term

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular') #padding
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

        self.embed = nn.Linear(c_in, int(1*d_model))
        # self.embed1 = nn.Linear(int(0.5*d_model), d_model)
        # self.relu = torch.nn.ReLU()
        # self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):

        # x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        x = self.embed(x)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class ValueEmbedding1(nn.Module):
    def __init__(self, c_in, d_model):
        super(ValueEmbedding1, self).__init__()

        #d_inp = freq_map[freq]
        self.embed = nn.Linear(c_in, d_model)

    def forward(self, x):
        return self.embed(x)

class EmergencyEmbedding(nn.Module):
    def __init__(self, emergency_label_length, d_model):
        super(EmergencyEmbedding, self).__init__()

        self.embed = nn.Linear(emergency_label_length, d_model)

    def forward(self, x):
        return self.embed(x)

# class EmergencyEmbedding(nn.Module):
#     def __init__(self, emergency_label_length, d_model):
#         super(EmergencyEmbedding, self).__init__()
#         self.tokenConv = nn.Conv1d(in_channels=1, out_channels=1,
#                                     kernel_size=3, padding=1, padding_mode='circular') #padding
#
#         self.tokenConv1 = nn.Conv2d(in_channels=1, out_channels=1,
#                                      kernel_size=3, padding=1, padding_mode='circular') #padding
#
#         self.tokenConv2 = nn.Conv2d(in_channels=1, out_channels=1,
#                                      kernel_size=5, padding= 2, stride = (1, 2), padding_mode='circular') #padding
#
#         self.embed = nn.Linear(emergency_label_length, 2*d_model)
#
#         # for markable day
#         # self.embed = nn.Linear(1, d_model)
#
#     def forward(self, x):
#
#         x = self.embed(x)
#         x = self.tokenConv2(torch.unsqueeze(x, 1))
#         x = x[:, 0,:, :]
#
#         return x#self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, emergency_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)


        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.emergency_anncouncement = EmergencyEmbedding(emergency_in, d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, x_emergency_mark):
        # t = x - x_mark
        # if torch.sum(t) == 0:
        #     x = self.value_embedding(x) + self.value_embedding1(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)  #+ self.value_embedding3(x3)
        # else:
        #     #x = self.position_embedding(x)+ self.temporal_embedding(x_mark) #+ self.value_embedding1(x)
        # for i in range(x_emergency_mark.shape[0]):
        #     x_emergency_mark[i] /= torch.max(x_emergency_mark[i]) if torch.max(x_emergency_mark[i]) != 0 else 1
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)# + self.emergency_anncouncement(x_emergency_mark)#   #
        # x = self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x), self.dropout(self.emergency_anncouncement(x_emergency_mark))

# class DataEmbedding1(nn.Module):
#     def __init__(self, c_in, emergency_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
#         super(DataEmbedding1, self).__init__()
#         self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
#         self.value_embedding1 = ValueEmbedding1(c_in=c_in, d_model=d_model)
#
#
#         self.position_embedding = PositionalEmbedding(d_model=d_model)
#         self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
#
#         self.emergency_anncouncement = EmergencyEmbedding(emergency_in, d_model=d_model)
#
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, x, x_mark, x_emergency_mark):
#         # t = x - x_mark
#         # if torch.sum(t) == 0:
#         #     x = self.value_embedding(x) + self.value_embedding1(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)  #+ self.value_embedding3(x3)
#         # else:
#         #     #x = self.position_embedding(x)+ self.temporal_embedding(x_mark) #+ self.value_embedding1(x)
#         # for i in range(x_emergency_mark.shape[0]):
#         #     x_emergency_mark[i] /= torch.max(x_emergency_mark[i]) if torch.max(x_emergency_mark[i]) != 0 else 1
#         x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)# + self.emergency_anncouncement(x_emergency_mark)#
#         # x = self.position_embedding(x) + self.temporal_embedding(x_mark)
#         return self.dropout(x)