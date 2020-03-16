import os
import sacrebleu


if __name__ == "__main__":
    DATA_DIR = "data/mscoco"
    PREDS_PATH = os.path.join(DATA_DIR, "preds_b256_tforcing_4_attn_layers_no_tanh_test_set_2_beam_5.txt")

    with open(PREDS_PATH, "r") as f_preds:
        hypotheses = [curr_hyp.strip() for curr_hyp in f_preds]

    refs1 = [[] for _ in range(3)]
    f_refs = [open(os.path.join(DATA_DIR, f"refs_b256_tforcing_4_attn_layers_no_tanh_test_set_2_{i}.txt")) for i in range(1, 4)]
    for i, f_curr_ref in enumerate(f_refs):
        for ref in f_curr_ref:
            refs1[i].append(ref.strip())

    for f in f_refs:
        f.close()

    refs2 = [[] for _ in range(4)]
    f_refs = [open(os.path.join(DATA_DIR, f"refs_b256_tforcing_4_attn_layers_no_tanh_test_set_2_{i}.txt")) for i in range(0, 4)]
    for i, f_curr_ref in enumerate(f_refs):
        for ref in f_curr_ref:
            refs2[i].append(ref.strip())

    for f in f_refs:
        f.close()

    assert len(refs1[0]) == len(refs1[1]) == len(refs1[2]) == 20_000
    assert len(refs2[0]) == len(refs2[1]) == len(refs2[2]) == len(refs2[3]) == 20_000

    bleu_3refs = sacrebleu.corpus_bleu(hypotheses, refs1)
    bleu_4refs = sacrebleu.corpus_bleu(hypotheses, refs2)

    print(f"mscoco_test_set_3(), 3 refs, SacreBLEU: {bleu_3refs}")
    print(f"mscoco_test_set_3(), 4 refs, SacreBLEU: {bleu_4refs}")
