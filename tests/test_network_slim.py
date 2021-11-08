import copy
import io
import json
import tempfile
import contextlib

import torch
from torch import nn
from loguru import logger
from pns import SlimPruner
from pns.tracker import gen_pruning_schema

from yolox.exp import get_exp


def random_init(M):
    for m in M.modules():
        try:
            if isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight.data)
            else:
                nn.init.kaiming_normal_(m.weight.data)
        except:
            pass


if __name__ == "__main__":
    exp_names = ["yolox-s", "yolox-m", "yolox-l", "yolox-x", "yolox-tiny"]

    pruning_ratios = [0.3, 0.4, 0.6, 0.7, 0.8]

    for name in exp_names:
        model = get_exp(None, name).get_model().eval()
        model.apply(random_init)

        x = torch.Tensor(1, 3, 224, 224)
        config = gen_pruning_schema(model, x)

        with tempfile.NamedTemporaryFile("w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.flush()
            for ratio in pruning_ratios:
                _model = copy.deepcopy(model)
                logger.info(f"Testing pruning [{name}] with ratio [{ratio}]")

                with contextlib.redirect_stdout(io.StringIO()):
                    pruner = SlimPruner(_model, f.name)
                    pruner.run(ratio)

                pruner.pruned_model.eval()
                x = torch.Tensor(1, 3, 224, 224)
                y = pruner.pruned_model(x)
