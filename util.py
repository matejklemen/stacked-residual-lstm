import torch
import torch.nn.functional as F


def encode_sequence(seq, forward_vocab):
    encoded = []
    for token in seq:
        encoded.append(forward_vocab.get(token, forward_vocab["<UNK>"]))
    return encoded


def decode_sequence(seq, backward_vocab):
    decoded = []
    for token in seq:
        decoded.append(backward_vocab[token])
    return decoded


def trim(encoded_seq, max_len):
    return encoded_seq[: max_len]


def pad(encoded_seq, max_len, pad_token):
    if len(encoded_seq) > max_len:
        raise ValueError(f"input sequence is longer than max_len: {len(encoded_seq)} > {max_len}")

    return encoded_seq + [pad_token] * (max_len - len(encoded_seq))


def greedy_decode(logits):
    # logits: (batch_size, seq_len, vocab_size)
    probas = F.softmax(logits, dim=2)
    best_idx = torch.argmax(probas, dim=2)
    return best_idx

