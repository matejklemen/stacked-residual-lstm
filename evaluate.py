import os
import sacrebleu


if __name__ == "__main__":
    DATA_DIR = "data/mscoco"
    MODEL_NAME = "4_layer_res13_attn1_lstm"
    model_load_dir = os.path.join(DATA_DIR, MODEL_NAME)
    PREDS_PATH = os.path.join(model_load_dir, "test_preds.txt")

    test_dir = os.path.join(DATA_DIR, "test_set")

    with open(PREDS_PATH, "r") as f_preds:
        hypotheses = [curr_hyp.strip() for curr_hyp in f_preds]

    refs1 = [[] for _ in range(4)]
    # convention: 0th reference is self-reference (if `include_self_ref` was set to True)
    f_refs = [open(os.path.join(test_dir, f"test_ref{i}.txt")) for i in range(1, 5)]
    for i, f_curr_ref in enumerate(f_refs):
        for ref in f_curr_ref:
            refs1[i].append(ref.strip())

    for f in f_refs:
        f.close()

    refs2 = [[] for _ in range(5)]
    f_refs = [open(os.path.join(test_dir, f"test_ref{i}.txt")) for i in range(0, 5)]
    for i, f_curr_ref in enumerate(f_refs):
        for ref in f_curr_ref:
            refs2[i].append(ref.strip())

    for f in f_refs:
        f.close()

    assert len(refs1[0]) == len(refs1[1]) == len(refs1[2]) == 20_000
    assert len(refs2[0]) == len(refs2[1]) == len(refs2[2]) == len(refs2[3]) == 20_000

    bleu_4refs = sacrebleu.corpus_bleu(hypotheses, refs1)
    bleu_5refs = sacrebleu.corpus_bleu(hypotheses, refs2)

    print(f"mscoco_test_set(), 4 refs, SacreBLEU: {bleu_4refs}")
    print(f"mscoco_test_set(), 5 refs, SacreBLEU: {bleu_5refs}")
