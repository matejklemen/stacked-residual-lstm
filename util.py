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


def encode_seq2seq(raw_set, forward_vocab, max_seq_len):
    """ Encodes (tokenized) source - target pairs or source singletons.
    Uses fixed special symbols `<PAD>`, `<BOS>`, `<EOS>` and `<UNK>`.

    Arguments
    ---------
    raw_set: list
        List of tokenized examples
    forward_vocab: dict
        Mapping from tokens to indices
    max_seq_len:
        Maximum sequence length (with included special tokens)

    Example
    -------
    >>> vocab = {"<PAD>": 0, "<BOS>": 1, ...}
    >>> pairs = [[["I", "am", "Iron", "Man"], ["My", "name", "is", "Iron", "Man"]]]
    >>> single = [[["My", "name", "is", "John", "Smith"]]]
    >>> encode_seq2seq(pairs, vocab, max_seq_len=10)
    """
    source_target = len(raw_set[0]) == 2
    encoded_source, source_lens = [], []
    if source_target:
        encoded_target, target_lens = [], []

    for example in raw_set:
        # source: <BOS> + sequence + <EOS>
        curr_src = trim(encode_sequence(example[0], forward_vocab), max_seq_len - 2)
        source_lens.append(len(curr_src) + 2)
        curr_src = pad([forward_vocab["<BOS>"]] + curr_src + [forward_vocab["<EOS>"]],
                       max_len=max_seq_len, pad_token=forward_vocab["<PAD>"])
        encoded_source.append(curr_src)

        # target: sequence + <EOS>
        if source_target:
            curr_tgt = trim(encode_sequence(example[1], forward_vocab), max_seq_len - 1)
            target_lens.append(len(curr_tgt) + 1)
            curr_tgt = pad(curr_tgt + [forward_vocab["<EOS>"]],
                           max_len=max_seq_len, pad_token=forward_vocab["<PAD>"])
            encoded_target.append(curr_tgt)

    encoded_source = torch.tensor(encoded_source, dtype=torch.long)
    source_lens = torch.tensor(source_lens, dtype=torch.long)
    if source_target:
        encoded_target = torch.tensor(encoded_target, dtype=torch.long)
        target_lens = torch.tensor(target_lens, dtype=torch.long)
        return encoded_source, source_lens, encoded_target, target_lens
    else:
        return encoded_source, source_lens


def greedy_decode(logits):
    # logits: (batch_size, seq_len, vocab_size)
    probas = F.softmax(logits, dim=2)
    best_idx = torch.argmax(probas, dim=2)
    return best_idx

