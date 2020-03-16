import os
import torch
import torch.nn.functional as F

from data_handler import load_vocab, mscoco_test_set_2, mscoco_test_set_3, mscoco_test_set_1
from seq2seq import ResidualLSTMEncoder, ResidualLSTMDecoder
from util import trim, pad, encode_sequence, decode_sequence


# NOTE: last encoder layer hidden states are shared between all states so no need to store them every time
class BeamState:
    def __init__(self, seq, logproba, curr_hids, curr_cells):
        self.seq = seq  # list of ints
        self.logproba = logproba  # float
        self.curr_hids = curr_hids
        self.curr_cells = curr_cells

    def __str__(self):
        return f"{' '.join(decode_sequence(self.seq, id2tok))} ({self.logproba / len(self.seq)})"


# copied from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def topp_sampling(logits, top_p=0.9, filter_value=-float("inf")):
    # logits: (batch_size, seq_len, vocab_size)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[..., indices_to_remove] = filter_value

    return logits


if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {device}")
    MAX_SEQ_LEN = 20
    TEST_BATCH_SIZE = 1  # predicting 1 example at a time (simplifies things)
    BEAM_SIZE = 5  # only applicable if using beam search
    TOP_P = 0.7  # only applicable if using nucleus sampling
    DATA_DIR = "data/mscoco"
    dev_name = "captions_val2014.json"
    dev_path = os.path.join(DATA_DIR, dev_name)
    tok2id, id2tok = load_vocab(os.path.join(DATA_DIR, "vocab.txt"))

    enc_model = ResidualLSTMEncoder(vocab_size=len(tok2id),
                                    num_layers=4, residual_layers=[1],
                                    inp_hid_size=512,
                                    dropout=0.5)
    dec_model = ResidualLSTMDecoder(vocab_size=len(tok2id),
                                    num_layers=4, residual_layers=[1],
                                    inp_size=512,
                                    hid_size=512,
                                    dropout=0.5,
                                    num_attn_layers=4)
    enc_model.load_state_dict(torch.load(os.path.join(DATA_DIR, "4_attn_layers_no_tanh_enc512.pt")))
    dec_model.load_state_dict(torch.load(os.path.join(DATA_DIR, "4_attn_layers_no_tanh_dec512.pt")))
    enc_model.to(device)
    dec_model.to(device)

    print("**Loaded model**")
    exit(0)

    """# Generating paraphrases for the test set"""
    raw_test_set, test_refs = mscoco_test_set_2(include_self_ref=True)
    test_input = []

    for curr_src, curr_tgt in raw_test_set:
        # source: <BOS> + sequence + <EOS>
        src = [tok2id["<BOS>"]] + trim(encode_sequence(curr_src, tok2id), MAX_SEQ_LEN - 2) + [tok2id["<EOS>"]]
        src = pad(src, MAX_SEQ_LEN, tok2id["<PAD>"])
        test_input.append(src)

    test_input = torch.tensor(test_input, dtype=torch.long)
    test_size = test_input.shape[0]

    print("**Encoded inputs**")

    """# Prediction using [nucleus (top p) sampling]"""

    # TEST_BATCH_SIZE = 1  # predicting 1 example at a time (simplifies things)
    # test_target = []
    # enc_model.eval()
    # dec_model.eval()
    # with torch.no_grad():
    #     for idx_ex in range(100):  # range(test_size):
    #         curr_src = test_input[idx_ex].unsqueeze(0).to(device)
    #         # encoder pass
    #         last_lay_hids, (last_t_hids, last_t_cells) = enc_model(curr_src)
    #
    #         curr_input = torch.tensor([[tok2id["<BOS>"]] for _ in range(TEST_BATCH_SIZE)],
    #                                   dtype=torch.long, device=device)
    #         curr_hids, curr_cells = last_t_hids, last_t_cells
    #
    #         decoded_seq = []
    #         # decoder pass
    #         for dec_step in range(MAX_SEQ_LEN):
    #             logits, dec_hids, dec_cells = dec_model(curr_input,
    #                                                     enc_hidden=last_lay_hids,
    #                                                     dec_hiddens=curr_hids,
    #                                                     dec_cells=curr_cells)
    #
    #             sampled_logits = topp_sampling(logits, top_p=0.7)
    #             probabilities = F.softmax(sampled_logits, dim=-1)
    #             curr_preds = torch.multinomial(probabilities.flatten(), 1).detach()
    #             curr_input = curr_preds.unsqueeze(1)
    #
    #             curr_token = curr_preds.cpu().item()
    #             if curr_token == tok2id["<EOS>"]:
    #                 break
    #             decoded_seq.append(id2tok[curr_token])
    #
    #         if (1 + idx_ex) % 500 == 0:
    #             print(f"**Generated targets for {1 + idx_ex} examples**")
    #         test_target.append(" ".join(decoded_seq))

    """# Prediction using [beam search]"""

    test_target = []
    enc_model.eval()
    dec_model.eval()
    with torch.no_grad():
        for idx_ex in range(test_size):
            remaining_beam = BEAM_SIZE
            curr_src = test_input[idx_ex].unsqueeze(0).to(device)

            # encoder pass
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
                    curr_input = torch.tensor([curr_state.seq[-1]],
                                              device=device).unsqueeze(0)
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
            prediction = max(finished, key=lambda s: s.logproba / len(s.seq)).seq[1: -1]
            test_target.append(" ".join([id2tok[i] for i in prediction]))

    # Need to save randomly selected examples (into 3 or 4 files, depending on total number of references)
    f_refs = [open(os.path.join(DATA_DIR, f"refs_b256_tforcing_4_attn_layers_no_tanh_test_set_2_{i}.txt"), "w") for i in range(4)]
    for curr_refs in test_refs:
        for i, r in enumerate(curr_refs):
            print(" ".join(r), file=f_refs[i])

    for f in f_refs:
        f.close()

    with open(os.path.join(DATA_DIR, f"preds_b256_tforcing_4_attn_layers_no_tanh_test_set_2_beam_{BEAM_SIZE}.txt"), "w") as f_preds:
        for curr_hyp in test_target:
            print(curr_hyp, file=f_preds)

    # for i, (curr_src, curr_tgt) in enumerate(raw_test_set[: len(test_target)]):
    #     print(f"[SOURCE] {' '.join(curr_src)}")
    #     print(f"[REFERENCE] {' '.join(curr_tgt)}")
    #     print(f"[HYPOTHESIS] {test_target[i]}")
    #     print("-------------------------------------------")


