import argparse
import json
import logging
import os

import torch
import torch.nn.functional as F

from data_handler import load_vocab, load_pairs
from seq2seq import ResidualLSTMEncoder, ResidualLSTMDecoder
from util import encode_seq2seq

parser = argparse.ArgumentParser(description="Generate sequences using a seq2seq model")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="data/mscoco",
                    help="A directory where data (source and target sequences) and a vocab file are assumed to be.")
parser.add_argument("--max_seq_len", type=int, default=20,
                    help="Max sequence length in number of tokens. <BOS> and <EOS> are included in this number.")
parser.add_argument("--beam_size", type=int, default=5)

pred_logger = logging.getLogger()
pred_logger.setLevel(logging.INFO)
DEFAULT_MODEL_DIR = "models/"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"Using device {DEVICE}")


# NOTE: last encoder layer hidden states are shared between all states so no need to store them every time
class BeamState:
    def __init__(self, seq, logproba, curr_hids, curr_cells):
        self.seq = seq  # list of ints
        self.logproba = logproba  # float
        self.curr_hids = curr_hids
        self.curr_cells = curr_cells


class Predictor:
    def __init__(self, model_name):
        self.model_load_dir = os.path.join(DEFAULT_MODEL_DIR, model_name)
        if not os.path.exists(self.model_load_dir):
            raise ValueError(f"Model could not be loaded from '{self.model_load_dir}'")

        config_path = os.path.join(self.model_load_dir, "config.json")
        with open(config_path) as f_config:
            logging.info(f"Loading model config from '{config_path}'")
            cfg = json.load(f_config)
            for attr_name, attr_value in cfg.items():
                logging.info(f"\t{attr_name} = {attr_value}")

        self.max_seq_len = cfg["max_seq_len"]

        self.tok2id, self.id2tok = load_vocab(os.path.join(self.model_load_dir, "vocab.txt"))
        logging.info(f"Loaded vocabulary with {len(self.tok2id)} tokens")

        self.enc_model = ResidualLSTMEncoder(vocab_size=len(self.tok2id),
                                             num_layers=cfg["num_layers"],
                                             residual_layers=cfg["residual_layers"],
                                             residual_n=cfg["residual_n"],
                                             inp_hid_size=cfg["enc_inp_hid_size"],
                                             dropout=cfg["dropout"],
                                             bidirectional=cfg["enc_bidirectional"]).to(DEVICE)
        self.dec_model = ResidualLSTMDecoder(vocab_size=len(self.tok2id),
                                             num_layers=cfg["num_layers"],
                                             residual_layers=cfg["residual_layers"],
                                             residual_n=cfg["residual_n"],
                                             inp_size=cfg["dec_inp_size"],
                                             hid_size=cfg["dec_hid_size"],
                                             dropout=cfg["dropout"],
                                             num_attn_layers=cfg["dec_attn_layers"]).to(DEVICE)

        enc_pretrained_path = os.path.join(self.model_load_dir, "enc.pt")
        dec_pretrained_path = os.path.join(self.model_load_dir, "dec.pt")
        if not os.path.exists(enc_pretrained_path) or not os.path.exists(dec_pretrained_path):
            raise ValueError(f"Trained encoder/decoder weights could not be found. Looked at "
                             f"'{enc_pretrained_path}' and '{dec_pretrained_path}'")
        self.enc_model.load_state_dict(torch.load(enc_pretrained_path))
        self.dec_model.load_state_dict(torch.load(dec_pretrained_path))
        self.enc_model.eval()
        self.dec_model.eval()
        logging.info(f"Loaded trained model")

    def run(self, test_src, **inference_kwargs):
        logging.info(f"Inference-time arguments: {inference_kwargs}")
        beam_size = inference_kwargs["beam_size"]
        max_seq_len = inference_kwargs.get("max_seq_len", self.max_seq_len)

        num_test = test_src.shape[0]
        test_target = []
        with torch.no_grad():
            for idx_ex in range(num_test):
                remaining_beam = beam_size
                curr_src = test_src[idx_ex].unsqueeze(0).to(DEVICE)

                last_lay_hids, (last_t_hids, last_t_cells) = self.enc_model(curr_src)
                curr_hids, curr_cells = last_t_hids, last_t_cells
                beam_states = [BeamState(seq=[self.tok2id["<BOS>"]],
                                         logproba=0.0,
                                         curr_hids=curr_hids,
                                         curr_cells=curr_cells)]
                finished = []

                # decoder pass
                for dec_step in range(max_seq_len):
                    curr_logprobas, curr_norm_logprobas = [], []
                    new_curr_hids, new_curr_cells = [], []
                    for curr_state in beam_states:
                        # extract and transform into shape the current decoder input
                        curr_input = torch.tensor([[curr_state.seq[-1]]], device=DEVICE)
                        curr_hids, curr_cells = curr_state.curr_hids, curr_state.curr_cells

                        logits, dec_hids, dec_cells = self.dec_model(curr_input,
                                                                     enc_hidden=last_lay_hids,
                                                                     dec_hiddens=curr_hids,
                                                                     dec_cells=curr_cells)

                        # logprobas of sequence ending with newly generated token
                        logprobas = F.log_softmax(logits[0][0], dim=0) + curr_state.logproba

                        # save newly generated candidate states
                        curr_logprobas.append(logprobas)
                        curr_norm_logprobas.append(logprobas / (len(curr_state.seq) + 1))
                        new_curr_hids.append(dec_hids)
                        new_curr_cells.append(dec_cells)

                    curr_logprobas = torch.cat(curr_logprobas)
                    curr_norm_logprobas = torch.cat(curr_norm_logprobas)

                    # get top candidate states
                    _, best_indices = torch.topk(curr_norm_logprobas, k=remaining_beam)

                    # create a new beam state (check if <EOS> and put in finished in that case)
                    for i in best_indices:
                        idx_seq, new_token = int(i) // len(self.tok2id), int(i) % len(self.tok2id)
                        prev_state = beam_states[idx_seq]
                        new_seq = prev_state.seq + [new_token]
                        new_hids, new_cells = new_curr_hids[idx_seq], new_curr_cells[idx_seq]

                        new_state = BeamState(new_seq, curr_logprobas[i],
                                              curr_hids=new_hids, curr_cells=new_cells)
                        if new_token == self.tok2id["<EOS>"]:
                            remaining_beam -= 1
                            finished.append(new_state)
                        else:
                            beam_states.append(new_state)

                    beam_states = beam_states[-remaining_beam:]
                    if len(beam_states) == 0:
                        break

                if beam_states:
                    for s in beam_states:
                        finished.append(s)

                if (1 + idx_ex) % 500 == 0:
                    print(f"**Generated targets for {1 + idx_ex} examples**")

                # best state, take sequence without BOS and EOS
                prediction = max(finished, key=lambda state: state.logproba / len(state.seq)).seq[1: -1]
                test_target.append(" ".join([self.id2tok[i] for i in prediction]))

        preds_path = os.path.join(self.model_load_dir, f"test_preds.txt")
        with open(preds_path, "w") as f_preds:
            for curr_hyp in test_target:
                print(curr_hyp, file=f_preds)
        logging.info(f"**Wrote predictions to '{preds_path}'**")


if __name__ == "__main__":
    args = parser.parse_args()
    tok2id, id2tok = load_vocab(os.path.join(DEFAULT_MODEL_DIR, args.model_name, "vocab.txt"))
    raw_test_set = load_pairs(src_path=os.path.join(args.data_dir, "test_set", "test_src.txt"),
                              tgt_path=os.path.join(args.data_dir, "test_set", "test_dst.txt"))
    test_input, _ = encode_seq2seq(raw_test_set, tok2id, args.max_seq_len)

    predictor = Predictor(model_name=args.model_name)
    predictor.run(test_input, max_seq_len=args.max_seq_len, beam_size=args.beam_size)

