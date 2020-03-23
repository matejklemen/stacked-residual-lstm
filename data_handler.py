import os
import json
import numpy as np
from nltk.tokenize import word_tokenize

PAD, BOS, EOS, UNK = 0, 1, 2, 3
MAX_TOKENS = 15

# TODO: move this inside if __name__ == "__main__"
DATA_DIR = "data/mscoco"
train_name = "captions_train2014.json"
dev_name = "captions_val2014.json"

train_path = os.path.join(DATA_DIR, train_name)
dev_path = os.path.join(DATA_DIR, dev_name)
np.random.seed(1)


def load_vocab(path):
    tok2id, id2tok = {}, {}

    # Fixed special tokens
    for idx, token in [(PAD, "<PAD>"), (BOS, "<BOS>"), (EOS, "<EOS>"), (UNK, "<UNK>")]:
        tok2id[token] = idx
        id2tok[idx] = token

    with open(path, "r") as f:
        for token in f:
            pr_token = token.lower().strip()
            if pr_token not in tok2id:
                tok2id[pr_token] = len(tok2id)
                id2tok[len(id2tok)] = pr_token

    return tok2id, id2tok


# Load and tokenize [source, target] pairs by path
def load_pairs(src_path, tgt_path):
    pairs = []
    with open(src_path) as f_src, open(tgt_path) as f_tgt:
        for curr_src in f_src:
            processed_src = curr_src.strip().split(" ")
            processed_tgt = f_tgt.readline().strip().split(" ")
            pairs.append([processed_src, processed_tgt])
    return pairs


def preprocess(seq):
    seq = seq.lower().replace(".", "").strip().replace("\n", " ")
    return word_tokenize(seq)[:MAX_TOKENS]


def read_image_annotations(path):
    # path... str (path to captions_train2014.json or captions_val2014.json file)
    with open(path) as f:
        data = json.load(f)

    id2caption = {}
    for annotation in data["annotations"]:
        img_id = annotation["image_id"]
        cap = annotation["caption"]
        if img_id in id2caption:
            id2caption[img_id].append(cap)
        else:
            id2caption[img_id] = [cap]

    return id2caption


def mscoco_training_set():
    dataset = []
    id2caption = read_image_annotations(train_path)
    for img_id, captions in id2caption.items():
        chosen_captions = np.random.choice(captions, size=4, replace=False).tolist()
        chosen_captions = list(map(preprocess, chosen_captions))

        src1, src2 = chosen_captions[0], chosen_captions[1]
        tgt1, tgt2 = chosen_captions[2], chosen_captions[3]

        dataset.append([src1, tgt1])
        dataset.append([tgt1, src1])
        dataset.append([src2, tgt2])
        dataset.append([tgt2, src2])

    return dataset


def mscoco_test_set(include_self_ref=False):
    dev_size, test_size = 20_000, 20_000
    dev_dataset, dev_refs = [], []
    test_dataset, test_refs = [], []
    id2caption = read_image_annotations(dev_path)
    assert len(id2caption) >= dev_size + test_size

    # Randomly select 20k images for each of dev and test set (non-overlapping) and create a
    # (1.) source-target pair (for loss-evaluation) and
    # (2.) source-references (for BLEU/METEOR/... evaluation)
    indices = np.random.choice(np.arange(len(id2caption)), size=(dev_size + test_size), replace=False)
    all_captions = list(id2caption.items())
    for idx in indices[: dev_size]:
        img_id, captions = all_captions[idx]
        chosen_captions = np.random.choice(captions, size=5, replace=False).tolist()
        chosen_captions = list(map(preprocess, chosen_captions))
        cap1, cap2, cap3, cap4, cap5 = chosen_captions

        dev_dataset.append([cap1, cap2])
        dev_refs.append([cap1, cap2, cap3, cap4, cap5] if include_self_ref else [cap2, cap3, cap4, cap5])

    for idx in indices[dev_size:]:
        img_id, captions = all_captions[idx]
        chosen_captions = np.random.choice(captions, size=5, replace=False).tolist()
        chosen_captions = list(map(preprocess, chosen_captions))
        cap1, cap2, cap3, cap4, cap5 = chosen_captions

        test_dataset.append([cap1, cap2])
        test_refs.append([cap1, cap2, cap3, cap4, cap5] if include_self_ref else [cap2, cap3, cap4, cap5])

    return (dev_dataset, dev_refs), (test_dataset, test_refs)


if __name__ == "__main__":
    train_dir = os.path.join(DATA_DIR, "train_set")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    # TODO: utility for obtaining & writing vocabulary of training set
    raw_train_set = mscoco_training_set()
    with open(os.path.join(train_dir, "train_src.txt"), "w") as f_src, \
            open(os.path.join(train_dir, "train_dst.txt"), "w") as f_dst:
        for curr_src, curr_tgt in raw_train_set:
            print(" ".join(curr_src), file=f_src)
            print(" ".join(curr_tgt), file=f_dst)
    print(f"**Wrote {len(raw_train_set)} train examples to '{train_dir}'**")

    dev_dir = os.path.join(DATA_DIR, "dev_set")
    if not os.path.exists(dev_dir):
        os.makedirs(dev_dir)

    test_dir = os.path.join(DATA_DIR, "test_set")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # NOTE: Self-reference is always placed as reference#0
    (raw_dev_set, _), (raw_test_set, test_refs) = mscoco_test_set(include_self_ref=True)

    with open(os.path.join(dev_dir, "dev_src.txt"), "w") as f_src, \
            open(os.path.join(dev_dir, "dev_dst.txt"), "w") as f_dst:
        # Only write source-target pairs for dev set
        for curr_src, curr_tgt in raw_dev_set:
            print(" ".join(curr_src), file=f_src)
            print(" ".join(curr_tgt), file=f_dst)
    print(f"**Wrote {len(raw_dev_set)} dev examples to '{dev_dir}'**")

    num_refs = len(test_refs[0])
    with open(os.path.join(test_dir, "test_src.txt"), "w") as f_src, \
            open(os.path.join(test_dir, "test_dst.txt"), "w") as f_dst:
        ref_files = [open(os.path.join(test_dir, f"test_ref{i}.txt"), "w") for i in range(num_refs)]
        # Write source-target pairs and source-references for test set
        for idx, (curr_src, curr_tgt) in enumerate(raw_test_set):
            curr_refs = test_refs[idx]
            print(" ".join(curr_src), file=f_src)
            print(" ".join(curr_tgt), file=f_dst)
            for idx_ref, ref in enumerate(curr_refs):
                print(" ".join(ref), file=ref_files[idx_ref])

        for f in ref_files:
            f.close()
    print(f"**Wrote {len(raw_test_set)} test examples to '{test_dir}', {num_refs} references per example**")
