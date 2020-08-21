import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ResidualLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, num_layers, residual_layers, input_size, hidden_size,
                 dropout=0.0, bidirectional=False, residual_n=1, padding_idx=0, pretrained_embs=None):
        super().__init__()

        self.is_res = np.zeros(num_layers, dtype=bool)
        if residual_layers:
            for layer_id in residual_layers:
                self.is_res[layer_id] = True
        self.residual_n = residual_n

        # internally, each direction will produce inp_hid_size/2 features, which will get concatenated back together
        if bidirectional and hidden_size % 2 != 0:
            raise ValueError("Hidden state size must be even")
        self.bidirectional = bidirectional
        num_directions = 2 if self.bidirectional else 1
        self.padding_idx = padding_idx

        self.input_size = input_size
        self.hidden_size = hidden_size // num_directions
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        if pretrained_embs is not None:
            self.embeddings = nn.Embedding.from_pretrained(pretrained_embs, padding_idx=padding_idx)
        else:
            self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                           embedding_dim=self.input_size,
                                           padding_idx=self.padding_idx)
        self.layers = []
        for idx_layer in range(num_layers):
            # Input to first LSTM are word embeddings, while input to other LSTMs are previous layer's hidden states
            curr_lstm_inp_size = self.input_size if idx_layer == 0 else self.hidden_size
            self.layers.append(nn.LSTM(input_size=curr_lstm_inp_size,
                                       hidden_size=self.hidden_size,
                                       batch_first=True,
                                       bidirectional=self.bidirectional))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, encoded_seq, seq_lens=None):
        """
        Arguments:
        ----------
        encoded_seq: torch.Tensor (batch_size, max_seq_len)
            Integer-encoded sequences in current batch
        seq_lens: torch.Tensor (batch_size)
            Sequence lengths of non-padded sequences in current batch

        Returns:
        --------
        torch.Tensor, (list, list):
            [0] torch.Tensor (batch_size, max_seq_len, hidden_size):
                hidden states of all timesteps for the last layer (for attention)
            [1] list:
                `num_layers` last hidden states for each layer (shape: (1, batch_size, hidden_size))
            [2] list:
                `num_layers` last cell states for each layer (shape: (1, batch_size, hidden_size))
        """
        # If data is padded beyond max sequence length inside current batch, trim it manually
        # (needed to ensure equal shapes of hidden states between LSTM layers)
        max_seq_len = seq_lens.max()
        trimmed_seq = encoded_seq[..., :max_seq_len]
        curr_inp = self.embeddings(trimmed_seq)

        last_t_hids, last_t_cells = [], []
        inputs_to_layers = [curr_inp.clone()]
        for i, curr_lstm in enumerate(self.layers):
            packed_input = pack_padded_sequence(curr_inp, lengths=seq_lens,
                                                batch_first=True, enforce_sorted=False)
            packed_out, (curr_hid, curr_cell) = curr_lstm(packed_input)
            (curr_out, _) = pad_packed_sequence(packed_out, batch_first=True)

            if i < self.num_layers - 1:
                curr_out = self.dropout(curr_out)

            inputs_to_layers.append(curr_out)
            # last_layer_hids = curr_out
            if self.is_res[i]:
                take_input_from = i - self.residual_n
                if take_input_from >= 0:
                    identity = inputs_to_layers[take_input_from]
                    # If residue vector is smaller, pad it, otherwise truncate it
                    if identity.shape[-1] < curr_out.shape[-1]:
                        identity = F.pad(identity, pad=[0, curr_out.shape[-1] - identity.shape[-1]])
                    else:
                        identity = identity[..., :curr_out.shape[-1]]
                else:
                    # e.g. there is no input from 5 layers back when we are at point after layer 0
                    # (the only input that could be added in this case is the embeddings)
                    raise ValueError(f"Cannot add identity from {self.residual_n} layers back after layer {i}")

                curr_inp = curr_out + identity
            else:
                curr_inp = curr_out

            last_t_hids.append(curr_hid)
            last_t_cells.append(curr_cell)

        return curr_inp, (last_t_hids, last_t_cells)


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
            # [Bahdanau (additive) attention]
            # Creates a custom representation of previous timestep hidden state
            self.dec_projector = nn.Linear(in_features=hid_size, out_features=hid_size, bias=False)
            # Creates a custom representation of encoder hidden states
            self.enc_projector = nn.Linear(in_features=hid_size, out_features=hid_size, bias=False)
            # Compresses the combined representation (obtained using the 2 lin. layers above) into unnormalized weights
            self.combinator = nn.Linear(in_features=hid_size, out_features=1, bias=False)
        elif num_attn_layers == 0:
            self.attn_layers = None
        else:
            raise ValueError(f"Valid options for 'num_attn_layers': 0 or 1")
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
        curr_inp = self.embeddings(encoded_input)  # [batch_size, 1, hidden_size]

        inputs_to_layers = []
        hids, cells = [], []
        for i, curr_lstm in enumerate(self.layers):
            if self.num_attn_layers == 1:
                # TODO: check everything's alright here
                combined_representation = F.tanh(self.dec_projector(dec_hiddens[i].permute(1, 0, 2)) +
                                                 self.enc_projector(enc_hidden))
                attn_weights = F.softmax(self.combinator(combined_representation), dim=1)
                weighted_comb = torch.sum(attn_weights * enc_hidden, dim=1)
                decoder_inp = torch.cat((curr_inp, weighted_comb), dim=2)
            else:  # no attention
                decoder_inp = curr_inp

            inputs_to_layers.append(decoder_inp)
            curr_out, (curr_hid, curr_cell) = curr_lstm(decoder_inp, (dec_hiddens[i], dec_cells[i]))

            if i < self.num_layers - 1:
                curr_out = self.dropout(curr_out)

            if self.is_res[i]:
                take_input_from = i - self.residual_n + 1
                if take_input_from >= 0:
                    identity = inputs_to_layers[take_input_from]
                else:
                    # e.g. there is no input from 5 layers back when we are at point after layer 0
                    # (the only input that could be added in this case is the input embeddings)
                    raise ValueError(f"Cannot add identity from {self.residual_n} layers back after layer {i}")
                curr_inp = curr_out + identity
            else:
                curr_inp = curr_out

            hids.append(curr_hid)
            cells.append(curr_cell)

        word_logits = self.fc(curr_inp)

        return word_logits, hids, cells
