import argparse
import json
import logging
import os
from datetime import datetime
from random import random
from time import time

import torch
import torch.nn as nn
import torch.optim as optim

from data_handler import load_vocab, load_pairs
from seq2seq import ResidualLSTMEncoder, ResidualLSTMDecoder
from util import encode_seq2seq, greedy_decode

parser = argparse.ArgumentParser(description="Train a seq2seq model")
parser.add_argument("--model_name", type=str, default=None)
# TODO: enable using these
parser.add_argument("--data_dir", type=str, default="data/mscoco",
                    help="A directory where data (source and target sequences) and a vocab file are assumed to be.")
# parser.add_argument("--vocab_file")

parser.add_argument("--max_seq_len", type=int, default=20,
                    help="Max sequence length in number of tokens. <BOS> and <EOS> are included in this number.")
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=4)

parser.add_argument("--log_every_n_batches", type=int, default=100)
parser.add_argument("--tf_proba_train", type=float, default=1.0,
                    help="Probability of using teacher forcing when training model on a batch.")
parser.add_argument("--tf_proba_dev", type=float, default=1.0,
                    help="Probability of using teacher forcing when validating model on a batch.")
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--residual_layers", type=str, default="1",
                    help="Comma-separated 0-based indices of layers, after which a residual connection is applied.")
parser.add_argument("--residual_n", type=int, default=1,
                    help="Input from how many layers back should be added in residual connection. "
                         "E.g. residual_n=1 means input of current layer gets added to the output of layer")
# TODO: separate encoder input and hidden size and perform dimensionality checks internally
parser.add_argument("--enc_inp_hid_size", type=int, default=512,
                    help="Input and hidden state size of the encoder model.")
parser.add_argument("--dec_inp_size", type=int, default=512)
parser.add_argument("--dec_hid_size", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.6)
parser.add_argument("--enc_bidirectional", action="store_true")
parser.add_argument("--dec_attn_layers", type=int, default=0,
                    help="Number of attention layers used when decoding. Supported values are None (no attention), "
                         "1 (same attention layer is used for all encoder layers) and num_layers (one attention layer "
                         "per encoder layer)")
parser.add_argument("--early_stopping_rounds", type=int, default=3)
parser.add_argument("--pretrained_embeddings", type=str, default="data/glove.6B.300d.txt")

train_logger = logging.getLogger()
train_logger.setLevel(logging.INFO)
DEFAULT_MODEL_DIR = "models/"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"Using device {DEVICE}")


def prepare_pretrained(path_to_embeddings, embedding_dim, vocab):
    embeddings = torch.randn((len(vocab), embedding_dim), dtype=torch.float32)
    embeddings[vocab["<PAD>"], :] = 0.0
    with open(path_to_embeddings, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            token = parts[0]
            vector = list(map(lambda s: float(s), parts[1:]))
            if token in vocab:
                embeddings[vocab[token], :] = torch.tensor(vector)

    norms = embeddings.norm(dim=1).unsqueeze(1)
    norms[vocab["<PAD>"], :] = 1.0
    embeddings /= norms

    return embeddings


class PerplexityLoss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=-100, reduction="mean"):
        super().__init__(ignore_index=ignore_index, reduction=reduction)

    def __call__(self, inp, target):
        # inp: (batch_size, vocab_size)
        # target: (batch_size)
        ce = super().__call__(inp, target)
        return torch.exp(ce)


class Trainer:
    def __init__(self, vocab, max_seq_len, num_epochs, batch_size, tf_proba_train, num_layers,
                 enc_inp_size, enc_hid_size, dec_inp_size, dec_hid_size, dropout, enc_bidirectional, dec_attn_layers,
                 model_name=None, tf_proba_dev=0.0, residual_layers=None, residual_n=1, early_stopping_rounds=1,
                 log_every_n_batches=100, pretrained_embs_path=None):
        self.model_name = datetime.now().strftime('%Y-%m-%dT%H:%M:%S') if model_name is None else model_name
        self.model_save_dir = os.path.join(DEFAULT_MODEL_DIR, self.model_name)
        print(f"Saving model details into '{self.model_save_dir}'")
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        self.max_seq_len = max_seq_len
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.tf_proba_train = tf_proba_train
        self.tf_proba_dev = tf_proba_dev
        # TODO: support for source vocab, target vocab?
        self.tok2id = vocab
        self.num_layers = num_layers
        self.residual_layers = residual_layers if residual_layers is not None else []
        self.residual_n = residual_n
        self.enc_inp_size = enc_inp_size
        self.enc_hid_size = enc_hid_size
        self.dec_inp_size = dec_inp_size
        self.dec_hid_size = dec_hid_size
        self.dropout = dropout
        self.enc_bidirectional = enc_bidirectional
        self.dec_attn_layers = dec_attn_layers
        self.log_every_n_batches = log_every_n_batches
        self.early_stopping_rounds = early_stopping_rounds

        assert self.batch_size > 1

        SKIP_CONFIG = {'model_save_dir', 'tok2id'}
        # Save config (= class attributes) before constructing the actual models
        with open(os.path.join(self.model_save_dir, "config.json"), "w") as f_config:
            config = {key: value for key, value in vars(self).items() if key not in SKIP_CONFIG}
            json.dump(config, fp=f_config, indent=4)
        logging.info("Model config:")
        for attr_name, attr_value in vars(self).items():
            if attr_name not in SKIP_CONFIG:
                logging.info(f"\t{attr_name} = {attr_value}")

        pretrained_embs = None
        if pretrained_embs_path is not None:
            pretrained_embs = prepare_pretrained(pretrained_embs_path, embedding_dim=self.enc_inp_size, vocab=vocab)

        self.enc_model = ResidualLSTMEncoder(vocab_size=len(self.tok2id),
                                             num_layers=self.num_layers,
                                             residual_layers=self.residual_layers,
                                             residual_n=residual_n,
                                             input_size=self.enc_inp_size,
                                             hidden_size=self.enc_hid_size,
                                             dropout=self.dropout,
                                             bidirectional=self.enc_bidirectional,
                                             pretrained_embs=pretrained_embs).to(DEVICE)
        self.dec_model = ResidualLSTMDecoder(vocab_size=len(self.tok2id),
                                             num_layers=self.num_layers,
                                             residual_layers=self.residual_layers,
                                             residual_n=residual_n,
                                             inp_size=self.dec_inp_size,
                                             hid_size=self.dec_hid_size,
                                             dropout=self.dropout,
                                             num_attn_layers=self.dec_attn_layers,
                                             pretrained_embs=pretrained_embs).to(DEVICE)

        self.optimizer = optim.SGD(list(self.enc_model.parameters()) + list(self.dec_model.parameters()), lr=1.0)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tok2id["<PAD>"])

        self.id2tok = {}
        for token, idx in self.tok2id.items():
            self.id2tok[idx] = token

        with open(os.path.join(self.model_save_dir, "vocab.txt"), "w") as f_vocab:
            # Note: assumes tokens are indexed from 0 to (|vocab| - 1)
            for token, _ in sorted(self.tok2id.items(), key=lambda tup: tup[1]):
                print(token, file=f_vocab)

        fh = logging.FileHandler(os.path.join(self.model_save_dir, 'train_log.info'))
        train_logger.addHandler(fh)

    def _process_batch(self, src, tgt, src_lens=None, tgt_lens=None, eval_mode=False):
        self.optimizer.zero_grad()

        curr_batch_size = src.shape[0]
        # Encoder pass
        last_lay_hids, (last_t_hids, last_t_cells) = self.enc_model(src, src_lens)

        curr_input = torch.tensor([[self.tok2id["<BOS>"]] for _ in range(curr_batch_size)],
                                  dtype=torch.long, device=DEVICE)
        curr_hids, curr_cells = last_t_hids, last_t_cells

        this_batch_logits = []
        teacher_forcing_proba = self.tf_proba_dev if eval_mode else self.tf_proba_train
        use_teacher_forcing = random() < teacher_forcing_proba
        # Decoder pass
        for dec_step in range(self.max_seq_len):
            logits, curr_hids, curr_cells = self.dec_model(curr_input,
                                                           enc_hidden=last_lay_hids,
                                                           dec_hiddens=curr_hids,
                                                           dec_cells=curr_cells)
            this_batch_logits.append(logits[:, 0, :].unsqueeze(2))
            curr_preds = greedy_decode(logits)

            if use_teacher_forcing:
                curr_input = tgt[:, dec_step].unsqueeze(1)
            else:
                curr_input = curr_preds

        # Always calculate loss - model selection metric
        batch_loss = self.loss_fn(torch.cat(this_batch_logits, dim=2), tgt)
        if not eval_mode:
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.enc_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.dec_model.parameters(), 1.0)
            self.optimizer.step()

        return float(batch_loss)

    def train(self, train_src, train_tgt, tr_src_lens=None, tr_tgt_lens=None):
        num_train_batches = (train_src.shape[0] + self.batch_size - 1) // self.batch_size
        num_batches_considered, train_loss = 0, 0.0
        shuffle_idx = torch.randperm(num_train_batches).to(DEVICE)

        self.enc_model.train()
        self.dec_model.train()
        for i, idx_batch in enumerate(shuffle_idx):
            start, end = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
            curr_src = train_src[start: end].to(DEVICE)
            curr_tgt = train_tgt[start: end].to(DEVICE)
            curr_batch_size = curr_src.shape[0]  # in case of partial batches
            num_batches_considered += curr_batch_size / self.batch_size

            train_loss += self._process_batch(curr_src, curr_tgt,
                                              tr_src_lens[start: end].to(DEVICE),
                                              tr_tgt_lens[start: end].to(DEVICE))

            if (1 + i) % self.log_every_n_batches == 0:
                train_logger.info(f"Batch #{1 + i}: train_loss={train_loss / num_batches_considered:.4f}")

        train_logger.info(f"Train_loss={train_loss / num_batches_considered:.4f}")
        self.scheduler.step()

        return train_loss / num_batches_considered

    def validate(self, dev_src, dev_tgt, dev_src_lens=None, dev_tgt_lens=None):
        num_dev_batches = (dev_src.shape[0] + self.batch_size - 1) // self.batch_size
        dev_loss, dev_batches_considered = 0.0, 0
        with torch.no_grad():
            self.enc_model.eval()
            self.dec_model.eval()
            for idx_batch in range(num_dev_batches):
                start, end = idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size
                curr_src = dev_src[start: end].to(DEVICE)
                curr_tgt = dev_tgt[start: end].to(DEVICE)
                curr_batch_size = curr_src.shape[0]  # in case of partial batches
                dev_batches_considered += curr_batch_size / self.batch_size

                dev_loss += self._process_batch(curr_src, curr_tgt,
                                                dev_src_lens[start: end].to(DEVICE),
                                                dev_tgt_lens[start: end].to(DEVICE),
                                                eval_mode=True)

        train_logger.info(f"Dev_loss={dev_loss / dev_batches_considered:.4f}")
        return dev_loss / dev_batches_considered

    def run(self, train_src, train_tgt, tr_src_lens=None, tr_tgt_lens=None,
            dev_src=None, dev_src_lens=None, dev_tgt=None, dev_tgt_lens=None):
        best_dev_loss, best_epoch = float("inf"), None

        t_start = time()
        for idx_epoch in range(self.num_epochs):
            train_logger.info(f"Epoch {1 + idx_epoch}")

            self.train(train_src, train_tgt, tr_src_lens=tr_src_lens, tr_tgt_lens=tr_tgt_lens)
            if dev_src is not None and dev_tgt is not None:
                # Save best state according to validation metric
                avg_dev_loss = self.validate(dev_src, dev_tgt, dev_src_lens=dev_src_lens, dev_tgt_lens=dev_tgt_lens)
                if avg_dev_loss < best_dev_loss:
                    train_logger.info(f"Saving new best model state to '{self.model_save_dir}'")
                    best_dev_loss, best_epoch = avg_dev_loss, idx_epoch
                    torch.save(self.enc_model.state_dict(), os.path.join(self.model_save_dir, "enc.pt"))
                    torch.save(self.dec_model.state_dict(), os.path.join(self.model_save_dir, "dec.pt"))
                if idx_epoch - best_epoch >= self.early_stopping_rounds:
                    train_logger.info(f"Stopping early because validation metric did not improve for "
                                      f"{self.early_stopping_rounds} rounds")
                    break
            else:
                # If no validation set is used, save last state
                torch.save(self.enc_model.state_dict(), os.path.join(self.model_save_dir, "enc.pt"))
                torch.save(self.dec_model.state_dict(), os.path.join(self.model_save_dir, "dec.pt"))

        train_logger.info(f"Best state was after epoch {1 + best_epoch}: loss = {best_dev_loss}")
        train_logger.info(f"Training took {time() - t_start}s")


# A utility function for checking learning rate decay
def log_lr(epoch, enc_opt, dec_opt):
    train_logger.info(f"Epoch #{epoch}:")
    train_logger.info(f"Encoder LR: {[group['lr'] for group in enc_opt.param_groups][0]}")
    train_logger.info(f"Decoder LR: {[group['lr'] for group in dec_opt.param_groups][0]}")


if __name__ == "__main__":
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    parsed_residual_layers = None
    # Turn comma-separated layer indices into a list
    if args.residual_layers:
        parsed_residual_layers = list(map(int, args.residual_layers.split(",")))
        for layer_id in parsed_residual_layers:
            if layer_id >= args.num_layers:
                raise ValueError(f"Cannot apply residual connection after non-existing layer "
                                 f"(max layer id allowed = num_layers - 1 = {args.num_layers - 1})")
        args.residual_layers = parsed_residual_layers

    torch.manual_seed(1)
    raw_train_set = load_pairs(src_path=os.path.join(DATA_DIR, "train_set", "train_src.txt"),
                               tgt_path=os.path.join(DATA_DIR, "train_set", "train_dst.txt"))#[: 1_000]
    raw_dev_set = load_pairs(src_path=os.path.join(DATA_DIR, "dev_set", "dev_src.txt"),
                             tgt_path=os.path.join(DATA_DIR, "dev_set", "dev_dst.txt"))#[: 200]
    tok2id, id2tok = load_vocab(os.path.join(DATA_DIR, "vocab.txt"))
    train_logger.info(f"{len(raw_train_set)} train examples, {len(raw_dev_set)} dev examples, vocab = {len(tok2id)} tokens")
    trainer = Trainer(model_name=args.model_name,
                      max_seq_len=args.max_seq_len,
                      num_epochs=args.num_epochs,
                      batch_size=args.batch_size,
                      tf_proba_train=args.tf_proba_train,
                      tf_proba_dev=args.tf_proba_dev,
                      num_layers=args.num_layers,
                      residual_layers=args.residual_layers,
                      residual_n=args.residual_n,
                      enc_inp_size=300,  # TODO: allow custom
                      enc_hid_size=args.enc_inp_hid_size,
                      dec_inp_size=300,  # TODO: allow custom
                      dec_hid_size=args.dec_hid_size,
                      dropout=args.dropout,
                      enc_bidirectional=args.enc_bidirectional,
                      dec_attn_layers=args.dec_attn_layers,
                      vocab=tok2id,  # TODO: add support for CLI use
                      log_every_n_batches=args.log_every_n_batches,
                      early_stopping_rounds=args.early_stopping_rounds,
                      pretrained_embs_path=args.pretrained_embeddings)

    train_input, tr_src_lens, train_target, tr_tgt_lens = encode_seq2seq(raw_train_set, tok2id, args.max_seq_len)
    dev_input, dev_src_lens, dev_target, dev_tgt_lens = encode_seq2seq(raw_dev_set, tok2id, args.max_seq_len)

    trainer.run(train_input, train_target, tr_src_lens=tr_src_lens, tr_tgt_lens=tr_tgt_lens,
                dev_src=dev_input, dev_tgt=dev_target, dev_src_lens=dev_src_lens, dev_tgt_lens=dev_tgt_lens)
