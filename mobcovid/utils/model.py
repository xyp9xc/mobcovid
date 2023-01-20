import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, emergency_in, c_out, seq_len, label_len, out_len,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.d_model = d_model
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding1 = DataEmbedding(enc_in, emergency_in, d_model, embed, freq, dropout)
        self.enc_embedding2 = DataEmbedding(enc_in, emergency_in, d_model, embed, freq, dropout)
        self.dec_embedding1 = DataEmbedding(dec_in, emergency_in, d_model, embed, freq, dropout)
        self.dec_embedding2 = DataEmbedding(dec_in, emergency_in, d_model, embed, freq, dropout)


        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder1 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.encoder2 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )



        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )


        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection1 = nn.Linear(d_model, c_out, bias=True)
        self.projection2 = nn.Linear(d_model, c_out, bias=True)




        #self.projection3 = nn.Linear(d_model, d_model, bias=True)

        self.emb = nn.Linear(d_model, d_model, bias=True)
        self.emb1 = nn.Linear(d_model, d_model, bias=True)
        self.emb2 = nn.Linear(d_model, d_model, bias=True)


        # att 8.25
        self.k_head = 2
        self.d_model = d_model
        self.q_1 = nn.Linear(d_model, d_model * self.k_head)
        self.k_1 = nn.Linear(d_model, d_model * self.k_head)
        self.v_1 = nn.Linear(d_model, d_model * self.k_head)

        self.q_2 = nn.Linear(d_model, d_model * self.k_head)
        self.k_2 = nn.Linear(d_model, d_model * self.k_head)
        self.v_2 = nn.Linear(d_model, d_model * self.k_head)
        self.gelu = nn.GELU()





        self.embed_s1 = nn.Linear(d_model, 1)
        self.embed_s2 = nn.Linear(d_model, 1)
        self.embed_s3 = nn.Linear(d_model, 1)
        self.embed_s4 = nn.Linear(d_model, 1)

        self.softmax = torch.nn.Softmax(dim=3)







        
    def forward(self, x_enc1, x_enc2, x_mark_enc1, x_mark_enc2, x_dec1, x_dec2, x_mark_dec, x_emergency_mark1, x_emergency_mark2, y_emergency_mark,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out1, emer_embed1 = self.enc_embedding1(x_enc1, x_mark_enc1, x_emergency_mark1)
        enc_out2, emer_embed2 = self.enc_embedding2(x_enc2, x_mark_enc2, x_emergency_mark2)

        # t1 = x_emergency_mark1[:, 3:, :]
        # t2 = x_emergency_mark2
        # t3 =  torch.mean(t2-t1)
        if x_enc1.shape[1] >= x_enc2.shape[1]:
            enc_out1, attns1 = self.encoder1(enc_out1+emer_embed1, attn_mask=enc_self_mask)
            enc_out2, attns2 = self.encoder2(enc_out2+emer_embed1[:, 0:x_enc2.shape[1], :], attn_mask=enc_self_mask)
        else:
            enc_out1, attns1 = self.encoder1(enc_out1+emer_embed2[:, 0:x_enc1.shape[1], :], attn_mask=enc_self_mask)
            enc_out2, attns2 = self.encoder2(enc_out2+emer_embed2, attn_mask=enc_self_mask)



        dec_out1, dec_emer1 = self.dec_embedding1(x_dec1, x_mark_dec, y_emergency_mark)
        dec_out2, dec_emer2 = self.dec_embedding2(x_dec2, x_mark_dec, y_emergency_mark)

        # dec_out1 = dec_out1 + dec_emer1
        # dec_out2 = dec_out2 + dec_emer2

        # dec_out1 = self.enc_embedding1(x_dec1, x_mark_dec, y_emergency_mark)
        # dec_out2 = self.enc_embedding2(x_dec2, x_mark_dec, y_emergency_mark)



        # enc_out22 = self.gelu(self.emb(torch.cat((enc_out2, enc_out_fix), 1)+enc_out1)) + torch.mul(torch.cat((enc_out2, enc_out_fix), 1),enc_out1)

        # enc_out22 = torch.mul(self.gelu(self.emb(torch.cat((enc_out2, enc_out_fix), 1))), enc_out1) \
        #             + torch.cat((enc_out2, enc_out_fix), 1)+ self.emb2(enc_out1)



        # enc_out22 = self.emb(torch.cat((enc_out2, enc_out_fix), 1)+enc_out1)


        # # att-based 8.25-xiaodan
        # enc_out_fix = torch.zeros(8,3,256).cuda()
        # enc_out_f2 = torch.cat((enc_out2, enc_out_fix), 1)
        # fea_q1 = self.q_1(enc_out1)
        # fea_q1 = fea_q1.reshape(fea_q1.size()[0], fea_q1.size()[1], self.k_head, self.d_model)
        #
        # fea_k1 = self.k_1(enc_out1)
        # fea_k1 = fea_k1.reshape(fea_k1.size()[0], fea_k1.size()[1], self.k_head, self.d_model)
        #
        # fea_v1 = self.v_1(enc_out1)
        # fea_v1 = fea_v1.reshape(fea_v1.size()[0], fea_v1.size()[1], self.k_head, self.d_model)
        #
        # fea_q2 = self.q_2(enc_out_f2)
        # fea_q2 = fea_q2.reshape(fea_q2.size()[0], fea_q2.size()[1], self.k_head, self.d_model)
        #
        # fea_k2 = self.k_2(enc_out_f2)
        # fea_k2 = fea_k2.reshape(fea_k2.size()[0], fea_k2.size()[1], self.k_head, self.d_model)
        #
        # fea_v2 = self.v_2(enc_out_f2)
        # fea_v2 = fea_v2.reshape(fea_v2.size()[0], fea_v2.size()[1], self.k_head, self.d_model)


        # att-based 8.25-xiaodan

        if x_enc1.shape[1] >= x_enc2.shape[1]:
            enc_out_fix = torch.zeros(8,x_enc1.shape[1]-x_enc2.shape[1], self.d_model).cuda()
            enc_out2 = torch.cat((enc_out2, enc_out_fix), 1)
        else:
            enc_out_fix = torch.zeros(8,x_enc2.shape[1]-x_enc1.shape[1], self.d_model).cuda()
            enc_out1 = torch.cat((enc_out1, enc_out_fix), 1)


        # enc_out_f2 = torch.cat((enc_out2, enc_out_fix), 1)


        fea_q1 = self.gelu(self.q_1(enc_out1))
        fea_q1 = fea_q1.reshape(fea_q1.size()[0], fea_q1.size()[1], self.k_head, self.d_model)

        fea_k1 = self.gelu(self.k_1(enc_out1))
        fea_k1 = fea_k1.reshape(fea_k1.size()[0], fea_k1.size()[1], self.k_head, self.d_model)

        fea_v1 = self.gelu(self.v_1(enc_out1))
        fea_v1 = fea_v1.reshape(fea_v1.size()[0], fea_v1.size()[1], self.k_head, self.d_model)

        fea_q2 = self.gelu(self.q_2(enc_out2))
        fea_q2 = fea_q2.reshape(fea_q2.size()[0], fea_q2.size()[1], self.k_head, self.d_model)

        fea_k2 = self.gelu(self.k_2(enc_out2))
        fea_k2 = fea_k2.reshape(fea_k2.size()[0], fea_k2.size()[1], self.k_head, self.d_model)

        fea_v2 = self.gelu(self.v_2(enc_out2))
        fea_v2 = fea_v2.reshape(fea_v2.size()[0], fea_v2.size()[1], self.k_head, self.d_model)



        # for 1
        fea_s1 = self.gelu(self.embed_s1(torch.mul(fea_q1, fea_k1)))
        fea_s2 = self.gelu(self.embed_s2(torch.mul(fea_q1, fea_k2)))

        fea_s = torch.cat((fea_s1,fea_s2), 3)
        fea_att = self.softmax(fea_s)

        fea_att1 = fea_att[:, :, :, 0:1].expand(fea_att.size()[0], fea_att.size()[1], fea_att.size()[2], self.d_model)
        fea_att2 = fea_att[:, :, :, 1:2].expand(fea_att.size()[0], fea_att.size()[1], fea_att.size()[2], self.d_model)


        fea1 = torch.mul(fea_v1, fea_att1) + torch.mul(fea_v2, fea_att2)


        # for 2
        fea_s1 = self.gelu(self.embed_s3(torch.mul(fea_q2, fea_k1)))
        fea_s2 = self.gelu(self.embed_s4(torch.mul(fea_q2, fea_k2)))


        fea_s = torch.cat((fea_s1,fea_s2), 3)
        fea_att = self.softmax(fea_s)

        fea_att1 = fea_att[:, :, :, 0:1].expand(fea_att.size()[0], fea_att.size()[1], fea_att.size()[2], self.d_model)
        fea_att2 = fea_att[:, :, :, 1:2].expand(fea_att.size()[0], fea_att.size()[1], fea_att.size()[2], self.d_model)

        fea2 = torch.mul(fea_v1, fea_att1) + torch.mul(fea_v2, fea_att2)

        enc_out22 = torch.sum(fea1, 2) + torch.sum(fea2, 2)



        dec_out22 = dec_out2 + dec_out1


        dec_out11 = self.decoder(dec_out22, enc_out22, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out22 = self.decoder(dec_out22, enc_out22, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        #dec_out11 = self.gelu(self.projection3(dec_out11)) + dec_out11

        dec_out1 = self.projection1(dec_out11)
        dec_out2 = self.projection2(dec_out22)
        #dec_out1 = dec_out2
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out1[:,-self.pred_len:,:], attns1, dec_out1[:,-self.pred_len:,:], attns2
        else:
            return dec_out1[:,-self.pred_len:,:], dec_out2[:,-self.pred_len:,:] # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
