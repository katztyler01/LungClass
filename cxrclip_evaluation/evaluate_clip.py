import glob
import json
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from cxrclip import seed_everything
from cxrclip.evaluator import Evaluator

log = logging.getLogger(__name__)


def print_evals(evals, metric="Accuracy(Micro)", best="max"):
    keys = list(evals.values())[0].keys()
    st = "| model | " + " | ".join(keys) + "|\n"
    st += "| :---- | " + " | ".join([("-" * (len(k) - 1)) + ":" for k in keys]) + "|\n"
    if best == "max":
        best_score = 0.0
    elif best == "min":
        best_score = 9e9
    else:
        raise ValueError("Unknown value for best, got %s" % best)
    best_ckpt = None
    for c, e in evals.items():
        filename = ".".join(c.split("/")[-1].split(".")[:-1])
        st += f"| {filename} | " + " | ".join([f"{e[k]:.3f}" for k in keys]) + " |\n"
        cur_score = e[metric]
        if best == "max":
            if best_score <= cur_score:
                best_score = cur_score
                best_ckpt = filename
        elif best == "min":
            if best_score >= cur_score:
                best_score = cur_score
                best_ckpt = filename
    st += f"Best {metric}: {best_score:.3f}, from {best_ckpt}\n"
    return st


@hydra.main(version_base=None, config_path="configs", config_name="eval_clip")
def main(cfg: DictConfig):
    print(cfg.test.checkpoint)
    seed_everything(cfg.test.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    OmegaConf.resolve(cfg)
    if type(cfg.test.checkpoint).__name__ == "ListConfig":
        ckpt_paths = cfg.test.checkpoint
    elif os.path.isdir(cfg.test.checkpoint):
        ckpt_paths = sorted(glob.glob(os.path.join(cfg.test.checkpoint, "*.tar")))
    else:
        ckpt_paths = sorted(glob.glob(cfg.test.checkpoint))

    print("ckpt_paths: ", ckpt_paths)
    cfg_dict = OmegaConf.to_container(cfg)
    evaluator = Evaluator(cfg_dict, ckpt_paths)
    save_path = os.path.dirname(ckpt_paths[0])
    print("EVALUATOR: ", evaluator.test_dataloader_dict.keys())
    for test_dataset_name in evaluator.test_dataloader_dict.keys():
        print(test_dataset_name)
        evals = {c: evaluator.evaluate_clip(c, test_dataset_name) for c in ckpt_paths}
        with open(os.path.join(save_path, f"results-{test_dataset_name}.json"), "w") as outfile:
            json.dump(evals, outfile)
        print("print best score")
        st = ""
        if test_dataset_name in {"chexpert5x200"}:
            st += "\nzeroshot_gloria\n"
            zeroshot_default = {k: v["zeroshot_gloria"]["mean/1000000"] for k, v in evals.items()}
            st += print_evals(zeroshot_default, metric="Accuracy(Micro)", best="max")

        if test_dataset_name in {"siim_pneumothorax", "vindr_cxr"}:
            st += "\nzeroshot binary\n"
            zeroshot_binary = {k: {_k: _v for _k, _v in v["zeroshot_binary"].items() if not isinstance(_v, dict)} for k, v in evals.items()}
            st += print_evals(zeroshot_binary, metric="AUROC(Avg)", best="max")

        if test_dataset_name in {"rsna_pneumonia"}:
            st += "\nZeroshot binary\n"
            zeroshot_binary = {k: {_k: _v for _k, _v in v["zeroshot_binary"].items() if not isinstance(_v, dict)} for k, v in evals.items()}
            st += print_evals(zeroshot_binary, metric="AUROC(Avg)", best="max")

        if test_dataset_name in {"chexpert5x200", "mimic_cxr", "openi"}:
            st += "\nI2T retrieval with deduplicated\n"
            retrieval_image_text = {k: v["retrieval_i2t"] for k, v in evals.items()}
            st += print_evals(retrieval_image_text, metric="MeanRank", best="min")

        log.info(cfg.test.checkpoint)
        log.info(st)

        with open(os.path.join(save_path, f"results-{test_dataset_name}.txt"), "w") as outfile:
            outfile.write(st)


if __name__ == "__main__":
    main()
