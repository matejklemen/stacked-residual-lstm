import os
import torch
import torch.nn.functional as F

from data_handler import load_vocab, load_pairs
from seq2seq import ResidualLSTMEncoder, ResidualLSTMDecoder
from util import encode_seq2seq, decode_sequence


# NOTE: last encoder layer hidden states are shared between all states so no need to store them every time
class BeamState:
    def __init__(self, seq, logproba, curr_hids, curr_cells):
        self.seq = seq  # list of ints
        self.logproba = logproba  # float
        self.curr_hids = curr_hids
        self.curr_cells = curr_cells

    def __str__(self):
        return f"{' '.join(decode_sequence(self.seq, id2tok))} ({self.logproba / len(self.seq)})"


if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {device}")

    DATA_DIR = "data/mscoco"
    MODEL_NAME = "4_layer_res13_attn1_lstm"
    model_load_dir = os.path.join(DATA_DIR, MODEL_NAME)

    MAX_SEQ_LEN = 20
    TEST_BATCH_SIZE = 1  # predicting 1 example at a time (simplifies things)
    BEAM_SIZE = 5  # only applicable if using beam search
    tok2id, id2tok = load_vocab(os.path.join(DATA_DIR, "vocab.txt"))

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
    enc_model.load_state_dict(torch.load(os.path.join(model_load_dir, "enc.pt")))
    dec_model.load_state_dict(torch.load(os.path.join(model_load_dir, "dec.pt")))

    raw_test_set = load_pairs(src_path=os.path.join(DATA_DIR, "test_set", "test_src.txt"),
                              tgt_path=os.path.join(DATA_DIR, "test_set", "test_dst.txt"))

    test_input, _ = encode_seq2seq(raw_test_set, tok2id, MAX_SEQ_LEN)
    num_test = test_input.shape[0]
    print(f"**Encoded {num_test} test examples**")

    test_target = []
    with torch.no_grad():
        enc_model.eval()
        dec_model.eval()
        for idx_ex in range(num_test):
            remaining_beam = BEAM_SIZE
            curr_src = test_input[idx_ex].unsqueeze(0).to(device)

            last_lay_hids, (last_t_hids, last_t_cells) = enc_model(curr_src)
            curr_hids, curr_cells = last_t_hids, last_t_cells
            beam_states = [BeamState(seq=[tok2id["<BOS>"]],
                                     logproba=0.0,
                                     curr_hids=curr_hids,
                                     curr_cells=curr_cells)]
            finished = []

            # decoder pass
            for dec_step in range(MAX_SEQ_LEN):
                curr_logprobas, curr_norm_logprobas = [], []
                new_curr_hids, new_curr_cells = [], []
                for curr_state in beam_states:
                    # extract and transform into shape the current decoder input
                    curr_input = torch.tensor([[curr_state.seq[-1]]], device=device)
                    curr_hids, curr_cells = curr_state.curr_hids, curr_state.curr_cells

                    logits, dec_hids, dec_cells = dec_model(curr_input,
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
                    idx_seq, new_token = i // len(tok2id), i % len(tok2id)
                    prev_state = beam_states[idx_seq]
                    new_seq = prev_state.seq + [new_token.item()]
                    new_hids, new_cells = new_curr_hids[idx_seq], new_curr_cells[idx_seq]

                    new_state = BeamState(new_seq, curr_logprobas[i],
                                          curr_hids=new_hids, curr_cells=new_cells)
                    if new_token == tok2id["<EOS>"]:
                        remaining_beam -= 1
                        finished.append(new_state)
                    else:
                        beam_states.append(new_state)
                beam_states = beam_states[-remaining_beam:]

            if beam_states:
                for s in beam_states:
                    finished.append(s)

            if (1 + idx_ex) % 500 == 0:
                print(f"**Generated targets for {1 + idx_ex} examples**")

            # best state, take sequence without BOS and EOS
            prediction = max(finished, key=lambda state: state.logproba / len(state.seq)).seq[1: -1]
            test_target.append(" ".join([id2tok[i] for i in prediction]))

    preds_path = os.path.join(model_load_dir, f"test_preds.txt")
    with open(preds_path, "w") as f_preds:
        for curr_hyp in test_target:
            print(curr_hyp, file=f_preds)
    print(f"**Wrote predictions to '{preds_path}'**")
