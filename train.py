import torch
import torch.optim as optim
import torch.nn as nn
import os

from time import time
from random import random

from data_handler import load_vocab, mscoco_training_set
from seq2seq import ResidualLSTMEncoder, ResidualLSTMDecoder
from util import trim, encode_sequence, pad, greedy_decode

torch.manual_seed(1)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device {device}")

DATA_DIR = "data/mscoco"
train_name = "captions_train2014.json"
dev_name = "captions_val2014.json"

train_path = os.path.join(DATA_DIR, train_name)
dev_path = os.path.join(DATA_DIR, dev_name)


class PerplexityLoss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=-100, reduction="mean"):
        super().__init__(ignore_index=ignore_index, reduction=reduction)

    def __call__(self, inp, target):
        # inp: (batch_size, vocab_size)
        # target: (batch_size)
        ce = super().__call__(inp, target)
        return torch.exp(ce)


MAX_SEQ_LEN = 20
raw_train_set = mscoco_training_set()
tok2id, id2tok = load_vocab(os.path.join(DATA_DIR, "vocab.txt"))
encoded_input, encoded_target = [], []

for curr_src, curr_tgt in raw_train_set:
    # source: <BOS> + sequence + <EOS>
    src = [tok2id["<BOS>"]] + trim(encode_sequence(curr_src, tok2id), MAX_SEQ_LEN - 2) + [tok2id["<EOS>"]]
    src = pad(src, MAX_SEQ_LEN, tok2id["<PAD>"])
    encoded_input.append(src)

    # target: sequence + <EOS>
    tgt = trim(encode_sequence(curr_tgt, tok2id), MAX_SEQ_LEN - 1) + [tok2id["<EOS>"]]
    tgt = pad(tgt, MAX_SEQ_LEN, tok2id["<PAD>"])
    encoded_target.append(tgt)

encoded_input = torch.tensor(encoded_input, dtype=torch.long)
encoded_target = torch.tensor(encoded_target, dtype=torch.long)


# SAMPLE_SIZE = 20_000
# rand_idx = torch.randperm(encoded_input.shape[0])[: SAMPLE_SIZE]
# encoded_input = encoded_input[rand_idx]
# encoded_target = encoded_target[rand_idx]

def log_lr(epoch, enc_opt, dec_opt):
    print(f"Epoch #{epoch}:")
    print(f"Encoder LR: {[group['lr'] for group in enc_optimizer.param_groups][0]}")
    print(f"Decoder LR: {[group['lr'] for group in dec_optimizer.param_groups][0]}")


def debug_preds(preds_tensor):
    # preds_tensor... (batch_size, max_seq_len)
    for idx_ex in range(min(4, preds_tensor.shape[0])):
        for idx_tok in range(MAX_SEQ_LEN):
            curr_tok = int(preds_tensor[idx_ex, idx_tok])
            print(id2tok[curr_tok], end=" ")
        print("")


# NOTE: originally, inp_size and hid_size (and inp_hid_size) are 512
enc_model = ResidualLSTMEncoder(vocab_size=len(tok2id),
                                num_layers=4, residual_layers=[1],
                                inp_hid_size=512,
                                dropout=0.5)
dec_model = ResidualLSTMDecoder(vocab_size=len(tok2id),
                                num_layers=4, residual_layers=[1],
                                inp_size=512,
                                hid_size=512,
                                dropout=0.5)

enc_model.to(device)
dec_model.to(device)

# SGD, learning rate = 1.0, halved after every 3rd epoch, trained for 10 epochs
enc_optimizer = optim.SGD(enc_model.parameters(), lr=1.0)
dec_optimizer = optim.SGD(dec_model.parameters(), lr=1.0)
enc_scheduler = optim.lr_scheduler.StepLR(enc_optimizer, step_size=3, gamma=0.5)
dec_scheduler = optim.lr_scheduler.StepLR(dec_optimizer, step_size=3, gamma=0.5)
ce_loss = PerplexityLoss(ignore_index=tok2id["<PAD>"])

NUM_EPOCHS = 10
BATCH_SIZE = 256
LOG_EVERY_N_BATCHES = 100
TEACHER_FORCING_PROBA = 1.0
assert BATCH_SIZE > 1

num_examples = encoded_input.shape[0]
batches_per_epoch = (num_examples + BATCH_SIZE - 1) // BATCH_SIZE
print(f"**{num_examples} examples, batch_size={BATCH_SIZE}, {batches_per_epoch} batches/epoch**")

t_start = time()
for idx_epoch in range(NUM_EPOCHS):
    print(f"Epoch {1 + idx_epoch}")
    log_lr(1 + idx_epoch, enc_optimizer, dec_optimizer)
    train_loss = 0.0
    shuffle_idx = torch.randperm(num_examples).to(device)
    num_batches_considered = 0

    enc_model.train()
    dec_model.train()

    for idx_batch in range(batches_per_epoch):
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        start = idx_batch * BATCH_SIZE
        end = (idx_batch + 1) * BATCH_SIZE
        curr_indices = shuffle_idx[start: end]
        num_batches_considered += curr_indices.shape[0] / BATCH_SIZE

        curr_src = encoded_input[curr_indices].to(device)
        curr_tgt = encoded_target[curr_indices].to(device)
        # this is needed to handle leftovers (not full batch)
        curr_batch_size = curr_src.shape[0]

        ## ---- ENCODER PASS -----------------------------------------------------
        last_lay_hids, (last_t_hids, last_t_cells) = enc_model(curr_src)

        # Experimental: zero out the hidden states corresponding to PAD tokens
        # TODO: if this is to be used, it needs to be copied (otherwise pytorch complains)
        # pad_mask = (curr_src == tok2id["<PAD>"])
        # last_lay_hids[pad_mask, :] = 0.0

        curr_input = torch.tensor([[tok2id["<BOS>"]] for _ in range(curr_batch_size)],
                                  dtype=torch.long, device=device)
        curr_hids, curr_cells = last_t_hids, last_t_cells

        this_batch_logits = []
        batch_preds = torch.zeros((curr_batch_size, MAX_SEQ_LEN), dtype=torch.long)
        batch_preds[:] = tok2id["<PAD>"]

        use_teacher_forcing = random() < TEACHER_FORCING_PROBA
        ## ---- DECODER PASS -----------------------------------------------------
        for dec_step in range(MAX_SEQ_LEN):
            # "Experimental": changed from logits, dec_hids, dec_cells = ...
            logits, curr_hids, curr_cells = dec_model(curr_input,
                                                      enc_hidden=last_lay_hids,
                                                      dec_hiddens=curr_hids,
                                                      dec_cells=curr_cells)
            this_batch_logits.append(logits[:, 0, :].unsqueeze(2))
            curr_preds = greedy_decode(logits)
            batch_preds[:, dec_step] = curr_preds.detach().squeeze(1)

            if use_teacher_forcing:
                curr_input = curr_tgt[:, dec_step].unsqueeze(1)
            else:
                curr_input = curr_preds

        batch_loss = ce_loss(torch.cat(this_batch_logits, dim=2), curr_tgt)
        train_loss += float(batch_loss)
        if (1 + idx_batch) % LOG_EVERY_N_BATCHES == 0:
            print(f"**Batch #{1 + idx_batch}: train_loss={train_loss / num_batches_considered:.4f}**")

        if (1 + idx_batch) % 100 == 0:
            debug_preds(batch_preds)

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(enc_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(dec_model.parameters(), 1.0)
        enc_optimizer.step()
        dec_optimizer.step()

    print(f"**Epoch #{1 + idx_epoch}: train_loss={train_loss / num_batches_considered:.4f}**")
    ## ---- SCHEDULING -------------------------------------------------------
    enc_scheduler.step()
    dec_scheduler.step()
print(f"**Training took {time() - t_start}s**")

"""# Loading and saving the model"""

# DATA_DIR
torch.save(enc_model.state_dict(), os.path.join(DATA_DIR, f"b{BATCH_SIZE}_4_attn_layers_no_tanh_enc512.pt"))
torch.save(dec_model.state_dict(), os.path.join(DATA_DIR, f"b{BATCH_SIZE}_4_attn_layers_no_tanh_dec512.pt"))

print(f"Saved model to {DATA_DIR}!")
