import argparse
import os
import sacrebleu
import logging

parser = argparse.ArgumentParser(description="Generate sequences using a seq2seq model")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--reference_files", type=str, required=True,
                    help="Comma-separated file names where references are stored - each file should contain one "
                         "example per line.")

eval_logger = logging.getLogger()
eval_logger.setLevel(logging.INFO)
DEFAULT_MODEL_DIR = "models/"


class Evaluator:
    def __init__(self, hyp_list, refs_lists, is_lowercased=False, save_path=None):
        self.hyp_list = hyp_list  # list of `num_examples` sequences
        self.refs_lists = refs_lists  # list of `num_ref_streams` lists, each containing `num_examples` sequences
        self.is_lowercased = is_lowercased
        self.save_path = save_path

        logging.info(f"Evaluating using {len(self.refs_lists)} reference streams")

    def run(self):
        sbleu = sacrebleu.corpus_bleu(self.hyp_list, self.refs_lists)
        if self.save_path is not None:
            with open(self.save_path) as f:
                print(sbleu, file=f)

        logging.info(f"[SacreBLEU] {sbleu}")
        return sbleu

    @staticmethod
    def from_files(hypotheses_path, references_paths, lowercase=False):
        with open(hypotheses_path) as f_hyp:
            hypotheses = [line.strip().lower() for line in f_hyp] if lowercase else [line.strip() for line in f_hyp]

        references = []
        for i, curr_ref_path in enumerate(references_paths, start=1):
            with open(curr_ref_path) as f_ref:
                curr_refs = [line.strip().lower() for line in f_ref] if lowercase else [line.strip() for line in f_ref]
                logging.info(f"Reference stream#{i}: {len(curr_refs)} references")
                references.append(curr_refs)

        return Evaluator(hypotheses, references, is_lowercased=lowercase)


if __name__ == "__main__":
    args = parser.parse_args()
    model_load_dir = os.path.join(DEFAULT_MODEL_DIR, args.model_name)
    PREDS_PATH = os.path.join(model_load_dir, "test_preds.txt")
    ref_paths = []
    for ref_fname in args.reference_files.split(","):
        ref_paths.append(ref_fname.strip())

    evaluator = Evaluator.from_files(hypotheses_path=PREDS_PATH, references_paths=ref_paths)
    evaluator.run()
