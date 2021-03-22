import torch
import torch.nn as nn
from emb import DataEmbedding, TokenEmbedding, InformerDataEmbedding
from layers import GroupLayer, ClassifierLayer, FullAttention, AttentionLayer
from encoder import Encoder, EncoderLayer, ConvLayer
from decoder import Decoder, DecoderLayer


class HLNet(nn.Module):
    def __init__(self, emb_in=1, seq_len=24, pred_len=8, emb_len=128, d_model=512, dropout=0.1,
                 group_factors=None, group_operator='avg', group_step=1, has_minute=False, has_hour=True):
        super(HLNet, self).__init__()

        if group_factors is None:
            group_factors = [8]
        else:
            group_factors = group_factors

        self.pred_len = pred_len

        # Encoding
        self.input_embedding = DataEmbedding(emb_in, emb_len, has_minute=has_minute, has_hour=has_hour)

        self.group_embeddings = nn.ModuleList([TokenEmbedding(emb_in, emb_len) for _ in group_factors])

        # Grouping prediction
        self.group_layers = nn.ModuleList([GroupLayer(gf, group_operator, group_step=group_step) for gf in group_factors])
        self.group_lstm_layers = nn.ModuleList(
            [nn.LSTM(emb_len, d_model) for i in range(len(group_factors))])

        self.group_outs = nn.ModuleList(
            [nn.Linear(d_model * (seq_len//group_step) * (i + 1), pred_len//group_step) for i, gf in enumerate(group_factors)])

        self.dropout = nn.Dropout(dropout)
        self.head_lstm = nn.LSTM(emb_len, d_model)
        
        out_in = d_model * seq_len + sum([d_model * (seq_len//group_step) * (i + 1) for i, gf in enumerate(group_factors)])

        self.out_layer = nn.Linear(out_in, pred_len, bias=True)

    def forward(self, inputs, input_mark):
        # input -> [batch_size, leq_len, features]

        embedding = self.input_embedding(inputs, input_mark)

        embedding = embedding.transpose(0, 1)  # [seq_len, batch_size, features]

        group_output = []
        previous_layer_hidden = None
        for group_layer, group_embedding, lstm_layer, group_out in zip(self.group_layers, self.group_embeddings, self.group_lstm_layers, self.group_outs):
            inputs_grouped = group_layer(inputs.transpose(0, 1))
            emb_grouped = group_embedding(inputs_grouped)  # [seq_len//group_factor, batch_size, features]

            group_hidden, _ = lstm_layer(emb_grouped)
            group_hidden = group_hidden.transpose(0, 1)

            group_hidden = self.dropout(torch.reshape(group_hidden, (len(group_hidden), -1)))  # Take the last hidden from last layer

            if previous_layer_hidden is not None:
                # The layers after must not interfere with previous layers
                hidden_in = torch.cat((group_hidden, previous_layer_hidden.clone().detach()), dim=1)
            else:
                hidden_in = group_hidden

            previous_layer_hidden = hidden_in  # Add at each level d_model to previous hidden layer
            x_out = group_out(hidden_in)
            group_output.append(x_out)

        head_out, _ = self.head_lstm(embedding)
        head_out = head_out.transpose(0,1)
        head_hidden = self.dropout(torch.reshape(head_out, (len(head_out), -1)))
        
        if previous_layer_hidden is not None:
            head_hidden = torch.cat((head_hidden, previous_layer_hidden.clone().detach()), dim=1)

        out = self.out_layer(head_hidden)

        return out, group_output

class HLNetOriginal(nn.Module):
    def __init__(self, emb_in=1, pred_len=8, emb_len=128, d_model=512, dropout=0.1,
                 group_factors=None, group_operator='avg', has_minute=False, has_hour=True):
        super(HLNet, self).__init__()

        if group_factors is None:
            group_factors = [8]
        else:
            group_factors = group_factors

        self.pred_len = pred_len

        # Encoding
        self.input_embedding = DataEmbedding(emb_in, emb_len, has_minute=has_minute, has_hour=has_hour)

        self.group_embeddings = nn.ModuleList([TokenEmbedding(emb_in, emb_len) for _ in group_factors])

        # Grouping prediction
        self.group_layers = nn.ModuleList([GroupLayer(gf, group_operator) for gf in group_factors])
        self.group_lstm_layers = nn.ModuleList(
            [nn.LSTM(emb_len, d_model) for i in range(len(group_factors))])

        self.group_outs = nn.ModuleList(
            [nn.Linear(d_model * (i + 1), pred_len // gf) for i, gf in enumerate(group_factors)])

        self.dropout = nn.Dropout(dropout)
        self.head_lstm = nn.LSTM(emb_len, d_model)

        self.out_layer = nn.Linear(d_model * (len(group_factors) + 1), pred_len, bias=True)

    def forward(self, inputs, input_mark):
        # input -> [batch_size, leq_len, features]

        embedding = self.input_embedding(inputs, input_mark)

        embedding = embedding.transpose(0, 1)  # [seq_len, batch_size, features]

        group_output = []
        previous_layer_hidden = None
        for group_layer, group_embedding, lstm_layer, group_out in zip(self.group_layers, self.group_embeddings, self.group_lstm_layers, self.group_outs):
            inputs_grouped = group_layer(inputs.transpose(0, 1))
            emb_grouped = group_embedding(inputs_grouped)  # [seq_len//group_factor, batch_size, features]
            # Maybe use same embedding than input?

            group_hidden, _ = lstm_layer(emb_grouped)
            group_hidden = group_hidden.transpose(0, 1)
            group_hidden = self.dropout(torch.reshape(group_hidden, (len(group_hidden, -1))))  # Take the last hidden from last layer

            if previous_layer_hidden is not None:
                # The layers after must not interfere with previous layers
                hidden_in = torch.cat((group_hidden, previous_layer_hidden.clone().detach()), dim=1)
            else:
                hidden_in = group_hidden

            previous_layer_hidden = hidden_in  # Add at each level d_model to previous hidden layer
            x_out = group_out(hidden_in)
            group_output.append(x_out)

        head_out, _ = self.head_lstm(embedding)
        head_out = head_out.transpose(0,1)
        head_hidden = self.dropout(torch.reshape(head_out, (len(head_out), -1)))
        
        if previous_layer_hidden is not None:
            head_hidden = torch.cat((head_hidden, previous_layer_hidden.clone().detach()), dim=1)
            
        out = self.out_layer(head_hidden)

        return out, group_output

class DACNet(nn.Module):
    def __init__(self, emb_in=1, pred_len=8, emb_len=128, d_model=512, dropout=0.1, k=2, has_minute=False, has_hour=True):
        super(DACNet, self).__init__()

        self.pred_len = pred_len
        self.d_model = d_model

        # Encoding
        self.input_embedding = DataEmbedding(emb_in, emb_len, has_minute=has_minute, has_hour=has_hour)
        self.classifier_embedding = nn.LSTM(emb_len, d_model)
        self.classifier = ClassifierLayer(d_model, k, 1.)

        self.heads = nn.ModuleList([nn.LSTM(d_model, d_model) for _ in range(k)])

        self.out_layer = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, inputs, input_mark):
        # input -> [batch_size, leq_len, features]

        embedding = self.input_embedding(inputs, input_mark)
        embedding = embedding.transpose(0, 1)  # [seq_len, batch_size, features]
        out, _ = self.classifier_embedding(embedding)

        hidden = out

        input_clusters = self.classifier(hidden[-1])
        input_clusters = torch.argmin(input_clusters, dim=1)
        self.input_clusters = input_clusters
        heads_hidden = torch.zeros((inputs.shape[0], self.d_model)).to('cuda:0').double()
        for cluster, head in enumerate(self.heads):
            cluster_hidden = hidden[:, input_clusters == cluster, :]
            if cluster_hidden.shape[1] > 1:
                out, _ = head(cluster_hidden)
                heads_hidden[input_clusters == cluster] = out[-1]

        out = self.out_layer(heads_hidden)

        return out

class BaselineLSTM(nn.Module):

    def __init__(self, emb_in=1, seq_len=24, pred_len=8, emb_len=128, d_model=512, dropout=0.1, n_layers=2, has_minute=False, has_hour=True):
        super(BaselineLSTM, self).__init__()

        self.pred_len = pred_len

        # Encoding
        self.embedding = DataEmbedding(emb_in, emb_len, has_minute=has_minute, has_hour=has_hour)

        self.lstm = nn.LSTM(emb_len, d_model, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

        self.out_layer = nn.Linear(d_model*seq_len, pred_len, bias=True)

    def forward(self, inputs, input_mark):
        # input -> [batch_size, leq_len, features]

        x = self.embedding(inputs, input_mark)

        x = x.transpose(0, 1)  # [seq_len, batch_size, features]
        x, _ = self.lstm(x)
        x = x.transpose(0, 1)
        x = self.dropout(torch.reshape(x,(len(x), -1)))

        out = self.out_layer(x)

        return out
        
class BaselineFC(nn.Module):

    def __init__(self, emb_in=1, pred_len=8, seq_len=24, emb_len=128, d_model=512, dropout=0.1, n_layers=2, has_minute=False, has_hour=True):
        super(BaselineFC, self).__init__()

        self.pred_len = pred_len

        # Encoding
        self.embedding = DataEmbedding(emb_in, emb_len, has_minute=has_minute, has_hour=has_hour)

        self.fcs = nn.ModuleList([nn.Linear(emb_len*seq_len if l==0 else d_model, d_model) for l in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        self.out_layer = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, inputs, input_mark):
        # input -> [batch_size, leq_len, features]

        x = self.embedding(inputs, input_mark)

        x = torch.reshape(x, (x.shape[0], -1))  # [seq_len, batch_size, features]
        for fc in self.fcs:
            x = fc(x)
        x = self.dropout(x)

        out = self.out_layer(x)

        return out
        

class HLInformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, group_factors=None,
                 group_operator='avg', group_step=1, dropout=0.0, attn='prob', embed='fixed', activation='gelu',
                 has_minute=False, has_hour=True):
        super(HLInformer, self).__init__()
        self.pred_len = out_len
        self.attn = attn

        if group_factors is None:
            group_factors = [4, 1]
        else:
            group_factors = [*group_factors, 1]

        self.group_factors = group_factors

        # Grouping
        self.group_layers = nn.ModuleList([GroupLayer(gf, group_operator, group_step) for gf in group_factors])
        # Encoding
        self.enc_embeddings = nn.ModuleList(
            [InformerDataEmbedding(enc_in, d_model, has_minute=has_minute, has_hour=has_hour) for _ in group_factors])
        self.dec_embeddings = nn.ModuleList(
            [InformerDataEmbedding(dec_in, d_model, has_minute=has_minute, has_hour=has_hour) for _ in group_factors])
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoders = nn.ModuleList([Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout),
                                   d_model, n_heads),
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
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        ) for _ in group_factors])
        # Decoder
        self.decoders = nn.ModuleList([Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        ) for _ in group_factors])
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projections = nn.ModuleList(
            [nn.Linear(d_model * (i + 1), c_out, bias=True) for i, gf in enumerate(group_factors)])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        group_projections = []
        last_dec_out = None
        last_dec_pred = None
        dec_out = None

        for i, (gf, group, enc_embedding, encoder, dec_embedding, decoder, projection) in enumerate(
                zip(self.group_factors, self.group_layers, self.enc_embeddings, self.encoders, self.dec_embeddings,
                    self.decoders, self.projections)):
            x_enc_grouped = group(x_enc, dim=1)
            
            if last_dec_pred is not None:
                x_dec[:, -self.pred_len:, :] = last_dec_pred
                
            dec_out = x_dec #group(x_dec, dim=1)

            if i == len(self.group_layers) - 1:
                enc_out = enc_embedding(x_enc_grouped, x_mark_enc)
                dec_out = dec_embedding(dec_out, x_mark_dec)

            else:
                enc_out = enc_embedding(x_enc_grouped)
                dec_out = dec_embedding(dec_out)

            enc_out = encoder(enc_out, attn_mask=enc_self_mask)

            dec_out = decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            #dec_out = dec_out.repeat_interleave(gf, dim=1)

            if last_dec_out is not None:
                # The layers after must not interfere with previous layers
                dec_out = torch.cat((dec_out, last_dec_out.clone().detach()), dim=-1)

            last_dec_out = dec_out

            dec_out = projection(dec_out)
            
            last_dec_pred = dec_out[:, -self.pred_len:, :].clone().detach()
            
            if i < len(self.group_layers) - 1:
                group_projections.append(dec_out[:, -self.pred_len:, :])

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        return dec_out[:, -self.pred_len::gf, :], group_projections



if __name__ == '__main__':
    model = BaselineLSTM()
    print(model(torch.zeros((32, 24, 1)), torch.zeros((32, 24, 4))).shape)

    model = HLNet(group_factors=[8, 4])
    output = model(torch.zeros((32, 24, 1)), torch.zeros((32, 24, 4)))
    print(output[0].shape)
    print(output[1][0].shape)
    print(output[1][1].shape)
