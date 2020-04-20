import torch
import torch.optim as optim
import torch.nn as nn
import os

from time import time
from random import random

from data_handler import load_vocab, load_pairs
from seq2seq import ResidualLSTMEncoder, ResidualLSTMDecoder
from util import encode_seq2seq, greedy_decode


class PerplexityLoss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=-100, reduction="mean"):
        super().__init__(ignore_index=ignore_index, reduction=reduction)

    def __call__(self, inp, target):
        # inp: (batch_size, vocab_size)
        # target: (batch_size)
        ce = super().__call__(inp, target)
        return torch.exp(ce)


# A utility function for checking learning rate decay
def log_lr(epoch, enc_opt, dec_opt):
    print(f"Epoch #{epoch}:")
    print(f"Encoder LR: {[group['lr'] for group in enc_opt.param_groups][0]}")
    print(f"Decoder LR: {[group['lr'] for group in dec_opt.param_groups][0]}")


# A utility function for prettyprinting a few predicted examples
def debug_preds(preds_tensor, id2tok):
    # preds_tensor... (batch_size, max_seq_len)
    # id2tok... mapping from integers to raw tokens
    for idx_ex in range(min(4, preds_tensor.shape[0])):
        for idx_tok in range(MAX_SEQ_LEN):
            curr_tok = int(preds_tensor[idx_ex, idx_tok])
            print(id2tok[curr_tok], end=" ")
        print("")


# A common train/dev function for running a single batch through the seq2seq model
def train_batch(batch_source, batch_target,
                enc_model, dec_model,
                enc_optimizer, dec_optimizer, loss_fn,
                teacher_forcing_proba, eval_mode=False):
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()

    curr_batch_size = batch_source.shape[0]
    last_lay_hids, (last_t_hids, last_t_cells) = enc_model(batch_source)

    curr_input = torch.tensor([[tok2id["<BOS>"]] for _ in range(curr_batch_size)],
                              dtype=torch.long, device=batch_source.device)
    curr_hids, curr_cells = last_t_hids, last_t_cells

    this_batch_logits = []
    use_teacher_forcing = random() < teacher_forcing_proba
    for dec_step in range(MAX_SEQ_LEN):
        logits, curr_hids, curr_cells = dec_model(curr_input,
                                                  enc_hidden=last_lay_hids,
                                                  dec_hiddens=curr_hids,
                                                  dec_cells=curr_cells)
        this_batch_logits.append(logits[:, 0, :].unsqueeze(2))
        curr_preds = greedy_decode(logits)

        if use_teacher_forcing:
            curr_input = batch_target[:, dec_step].unsqueeze(1)
        else:
            curr_input = curr_preds

    batch_loss = loss_fn(torch.cat(this_batch_logits, dim=2), batch_target)
    if not eval_mode:
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(enc_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(dec_model.parameters(), 1.0)
        enc_optimizer.step()
        dec_optimizer.step()

    return float(batch_loss)


if __name__ == "__main__":
    MAX_SEQ_LEN = 20
    NUM_EPOCHS = 30
    BATCH_SIZE = 256
    LOG_EVERY_N_BATCHES = 100
    TEACHER_FORCING_PROBA = 1.0
    assert BATCH_SIZE > 1

    DATA_DIR = "data/mscoco"
    MODEL_NAME = "4_layer_res13_attn1_lstm"
    model_save_dir = os.path.join(DATA_DIR, MODEL_NAME)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    torch.manual_seed(1)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"**Using device {device}**")

    raw_train_set = load_pairs(src_path=os.path.join(DATA_DIR, "train_set", "train_src.txt"),
                               tgt_path=os.path.join(DATA_DIR, "train_set", "train_dst.txt"))
    raw_dev_set = load_pairs(src_path=os.path.join(DATA_DIR, "dev_set", "dev_src.txt"),
                             tgt_path=os.path.join(DATA_DIR, "dev_set", "dev_dst.txt"))
    tok2id, id2tok = load_vocab(os.path.join(DATA_DIR, "vocab.txt"))
    print(f"**{len(raw_train_set)} train examples, {len(raw_dev_set)} dev examples, vocab = {len(tok2id)} tokens**")

    train_input, train_target = encode_seq2seq(raw_train_set, tok2id, MAX_SEQ_LEN)
    dev_input, dev_target = encode_seq2seq(raw_dev_set, tok2id, MAX_SEQ_LEN)

    enc_model = ResidualLSTMEncoder(vocab_size=len(tok2id),
                                    num_layers=4, residual_layers=[1, 3],
                                    inp_hid_size=512,
                                    dropout=0.5,
                                    bidirectional=False).to(device)
    dec_model = ResidualLSTMDecoder(vocab_size=len(tok2id),
                                    num_layers=4, residual_layers=[1, 3],
                                    inp_size=512,
                                    hid_size=512,
                                    dropout=0.5,
                                    num_attn_layers=1).to(device)

    # SGD, learning rate = 1.0, halved after every 3rd epoch, trained for 10 epochs
    enc_optimizer = optim.SGD(enc_model.parameters(), lr=1.0)
    dec_optimizer = optim.SGD(dec_model.parameters(), lr=1.0)
    enc_scheduler = optim.lr_scheduler.StepLR(enc_optimizer, step_size=3, gamma=0.5)
    dec_scheduler = optim.lr_scheduler.StepLR(dec_optimizer, step_size=3, gamma=0.5)
    loss_fn = PerplexityLoss(ignore_index=tok2id["<PAD>"])

    best_dev_loss, best_epoch = float("inf"), None
    num_train, num_dev = train_input.shape[0], dev_input.shape[0]
    train_batches = (num_train + BATCH_SIZE - 1) // BATCH_SIZE
    dev_batches = (num_dev + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"**{num_train} train examples, {num_dev} dev examples, "
          f"batch_size={BATCH_SIZE}, {train_batches} train batches/epoch**")

    t_start = time()
    for idx_epoch in range(NUM_EPOCHS):
        log_lr(1 + idx_epoch, enc_optimizer, dec_optimizer)
        num_batches_considered, train_loss = 0, 0.0
        shuffle_idx = torch.randperm(train_batches).to(device)

        enc_model.train()
        dec_model.train()
        for idx_batch in shuffle_idx:
            start, end = idx_batch * BATCH_SIZE, (idx_batch + 1) * BATCH_SIZE
            curr_src = train_input[start: end].to(device)
            curr_tgt = train_target[start: end].to(device)
            curr_batch_size = curr_src.shape[0]  # in case of partial batches
            num_batches_considered += curr_batch_size / BATCH_SIZE

            train_loss += train_batch(curr_src, curr_tgt, enc_model, dec_model, enc_optimizer, dec_optimizer, loss_fn,
                                      teacher_forcing_proba=TEACHER_FORCING_PROBA)

            if (1 + idx_batch) % LOG_EVERY_N_BATCHES == 0:
                print(f"**Batch #{1 + idx_batch}: train_loss={train_loss / num_batches_considered:.4f}**")

        print(f"**Epoch #{1 + idx_epoch}: train_loss={train_loss / num_batches_considered:.4f}**")
        enc_scheduler.step()
        dec_scheduler.step()

        dev_loss, dev_batches_considered = 0.0, 0
        # Evaluation on dev set after each epoch
        with torch.no_grad():
            enc_model.eval()
            dec_model.eval()
            for idx_batch in range(dev_batches):
                start, end = idx_batch * BATCH_SIZE, (idx_batch + 1) * BATCH_SIZE
                curr_src = dev_input[start: end].to(device)
                curr_tgt = dev_target[start: end].to(device)
                curr_batch_size = curr_src.shape[0]  # in case of partial batches
                dev_batches_considered += curr_batch_size / BATCH_SIZE

                dev_loss += train_batch(curr_src, curr_tgt, enc_model, dec_model, enc_optimizer, dec_optimizer, loss_fn,
                                        teacher_forcing_proba=TEACHER_FORCING_PROBA if TEACHER_FORCING_PROBA > (1.0 - 1e-5) else 0.0,
                                        eval_mode=True)

        curr_dev_loss = dev_loss / dev_batches_considered
        print(f"**Epoch #{1 + idx_epoch}: dev_loss={curr_dev_loss:.4f}**")
        if curr_dev_loss < best_dev_loss:
            print(f"**Saving new best model state to '{model_save_dir}'**")
            best_dev_loss, best_epoch = curr_dev_loss, idx_epoch
            torch.save(enc_model.state_dict(), os.path.join(model_save_dir, "enc.pt"))
            torch.save(dec_model.state_dict(), os.path.join(model_save_dir, "dec.pt"))

    print(f"**Best state was after epoch {1 + best_epoch}: loss = {best_dev_loss}**")
    print(f"**Training took {time() - t_start}s**")
