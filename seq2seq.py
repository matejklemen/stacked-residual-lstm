import numpy as np
import torch
import torch.nn as nn
from torchnlp.nn import Attention


class ResidualLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, num_layers, residual_layers,
                 inp_hid_size, dropout=0.0, bidirectional=False, residual_n=1):
        super().__init__()

        self.is_res = np.zeros(num_layers, dtype=bool)
        if residual_layers:
            for layer_id in residual_layers:
                self.is_res[layer_id] = True
        self.residual_n = residual_n

        # internally, each direction will produce inp_hid_size/2 features, which will get concatenated back together
        if bidirectional and inp_hid_size % 2 != 0:
            raise ValueError("Hidden state size must be even")
        self.bidirectional = bidirectional
        num_directions = 2 if self.bidirectional else 1

        self.input_size = inp_hid_size
        self.hidden_size = inp_hid_size // num_directions
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=self.input_size)
        self.layers = nn.ModuleList([nn.LSTM(input_size=self.input_size,
                                             hidden_size=self.hidden_size,
                                             batch_first=True,
                                             bidirectional=self.bidirectional) for _ in range(num_layers)])

    def forward(self, encoded_seq):
        """
        Arguments:
        ----------
        encoded_seq: torch.Tensor
            (batch_size, max_seq_len) tensor, containing integer-encoded
            sequences in current batch

        Returns:
        --------
        torch.Tensor, (list, list):
            [0] hidden states of all timesteps for the last layer (for attention)
                ((batch_size, max_seq_len, hidden_size) tensor)
            [1] list of last hidden states for each layer
                (num_layers * (1, batch_size, hidden_size) tensors)
            [2] list of last cell states for each layer
                (num_layers * (1, batch_size, hidden_size) tensors)
        """
        embedded = self.dropout(self.embeddings(encoded_seq))

        last_t_hids, last_t_cells = [], []
        all_layers_hids = [embedded]
        curr_inp = embedded
        for i, curr_lstm in enumerate(self.layers):
            curr_out, (curr_hid, curr_cell) = curr_lstm(curr_inp)

            all_layers_hids.append(curr_out)
            # last_layer_hids = curr_out
            if self.is_res[i]:
                take_input_from = i - self.residual_n + 1
                if take_input_from >= 0:
                    identity = all_layers_hids[take_input_from]
                else:
                    # e.g. there is no input from 5 layers back when we are at point after layer 0
                    # (the only input that could be added in this case is the embeddings)
                    raise ValueError(f"Cannot add identity from {self.residual_n} layers back after layer {i}")

                curr_inp = curr_out + identity
            else:
                curr_inp = curr_out

            # after all but last layer
            if i < self.num_layers - 1:
                curr_inp = self.dropout(curr_inp)

            last_t_hids.append(curr_hid)
            last_t_cells.append(curr_cell)

        return all_layers_hids[-1], (last_t_hids, last_t_cells)


class ResidualLSTMDecoder(nn.Module):
    def __init__(self, vocab_size, num_layers, residual_layers,
                 inp_size, hid_size, dropout=0.0, num_attn_layers=1, residual_n=1):
        super().__init__()

        self.is_res = np.zeros(num_layers, dtype=bool)
        if residual_layers:
            for layer_id in residual_layers:
                self.is_res[layer_id] = True
        self.residual_n = residual_n

        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=inp_size)
        if num_attn_layers == 1:
            self.attn_layers = Attention(hid_size)
        elif num_attn_layers == num_layers:
            self.attn_layers = nn.ModuleList([Attention(hid_size) for _ in range(num_layers)])
        elif num_attn_layers == 0:
            self.attn_layers = None
        else:
            raise ValueError(f"Valid options for 'num_attn_layers': 0 or 1 or {num_layers} (= num. LSTM layers)")
        self.num_attn_layers = num_attn_layers

        # LSTMs will get 2 things as input (IF using attention):
        # [attended encoder states, embedded decoder input/hidden state] (concatenated)
        self.layers = nn.ModuleList([nn.LSTM(input_size=(inp_size + hid_size) if num_attn_layers > 0 else inp_size,
                                             hidden_size=hid_size,
                                             batch_first=True) for _ in range(num_layers)])
        self.fc = nn.Linear(hid_size, vocab_size)

    def forward(self, encoded_input, enc_hidden,
                dec_hiddens, dec_cells):
        # encoded_input: (batch_size, 1) tensor
        # enc_hidden: ... (batch_size, max_seq_len, hidden_size) tensor
        # dec_hiddens: ... list of num_layers * (1, batch_size, hidden_size) tensors
        # dec_cells: ... list of num_layers * (1, batch_size, hidden_size) tensors
        embedded_input = self.dropout(self.embeddings(encoded_input))
        curr_inp = embedded_input

        all_layers_hids = []
        hids, cells = [], []
        for i, curr_lstm in enumerate(self.layers):
            if self.num_attn_layers == 1:
                weighted_comb, _ = self.attn_layers(query=dec_hiddens[i].transpose(0, 1), context=enc_hidden)
                decoder_inp = torch.cat((weighted_comb, curr_inp), dim=2)
            elif self.num_attn_layers == self.num_layers:
                weighted_comb, _ = self.attn_layers[i](query=dec_hiddens[i].transpose(0, 1), context=enc_hidden)
                decoder_inp = torch.cat((weighted_comb, curr_inp), dim=2)
            else:  # no attention
                decoder_inp = curr_inp

            all_layers_hids.append(decoder_inp)
            curr_out, (curr_hid, curr_cell) = curr_lstm(decoder_inp, (dec_hiddens[i], dec_cells[i]))
            if self.is_res[i]:
                take_input_from = i - self.residual_n + 1
                if take_input_from >= 0:
                    identity = all_layers_hids[take_input_from]
                else:
                    # e.g. there is no input from 5 layers back when we are at point after layer 0
                    # (the only input that could be added in this case is the input embeddings)
                    raise ValueError(f"Cannot add identity from {self.residual_n} layers back after layer {i}")
                curr_inp = curr_out + identity
            else:
                curr_inp = curr_out

            # after all but last layer
            if i < self.num_layers - 1:
                curr_inp = self.dropout(curr_inp)

            hids.append(curr_hid)
            cells.append(curr_cell)

        word_logits = self.fc(curr_inp)

        return word_logits, hids, cells
