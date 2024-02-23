import torch
import torch.nn as nn

from models.embed import DataEmbedding


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.input_embedding = DataEmbedding(input_dim, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        # 输出的特征维度
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src_x, src_x_mark, tgt_x, tgt_x_mark, tgt_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        src = self.input_embedding(src_x, src_x_mark)
        tgt = self.input_embedding(tgt_x, tgt_x_mark)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
        output = self.output_layer(output)
        return output


