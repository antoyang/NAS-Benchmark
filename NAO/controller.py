import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder


SOS_ID = 0
EOS_ID = 0
INITRANGE=0.04

class NAO(nn.Module):
    def __init__(self,
                 encoder_layers,
                 encoder_vocab_size,
                 encoder_hidden_size,
                 encoder_dropout,
                 encoder_length,
                 source_length,
                 encoder_emb_size,
                 mlp_layers,
                 mlp_hidden_size,
                 mlp_dropout,
                 decoder_layers,
                 decoder_vocab_size,
                 decoder_hidden_size,
                 decoder_dropout,
                 decoder_length,
                 ):
        super(NAO, self).__init__()
        self.encoder = Encoder(
            encoder_layers,
            encoder_vocab_size,
            encoder_hidden_size,
            encoder_dropout,
            encoder_length,
            source_length,
            encoder_emb_size,
            mlp_layers,
            mlp_hidden_size,
            mlp_dropout,
        )
        self.decoder = Decoder(
            decoder_layers,
            decoder_vocab_size,
            decoder_hidden_size,
            decoder_dropout,
            decoder_length,
            encoder_length
        )

        self.init_parameters()
        self.flatten_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.uniform_(w.data, -INITRANGE, INITRANGE)
    
    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
    
    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self.encoder(input_variable)
        decoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder(target_variable, decoder_hidden, encoder_outputs)
        decoder_outputs = torch.stack(decoder_outputs, 0).permute(1, 0, 2)
        arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return predict_value, decoder_outputs, arch
    
    def generate_new_arch(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb = self.encoder.infer(
            input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = (new_arch_emb.unsqueeze(0), new_arch_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder(None, new_encoder_hidden, new_encoder_outputs)
        new_arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return new_arch
