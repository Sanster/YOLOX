import argparse
import io
import json
import tempfile
import contextlib

import torch
from loguru import logger
from pns import SlimPruner
from pns.tracker import gen_pruning_schema

from yolox.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="save path of schema file",
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    model = get_exp(args.exp_file, args.name).get_model().eval()

    x = torch.Tensor(1, 3, 224, 224)
    config = gen_pruning_schema(model, x)

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.flush()

        for ratio in [0.6, 0.7, 0.8, 0.9]:
            logger.info(f"Testing pruning [{args.name}] with ratio [{ratio}]")

            with contextlib.redirect_stdout(io.StringIO()):
                pruner = SlimPruner(model, f.name)
                pruner.run(ratio)

            pruner.pruned_model.eval()
            x = torch.Tensor(1, 3, 224, 224)
            y = pruner.pruned_model(x)
