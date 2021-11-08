import torch
from torch import nn
from pns import SlimPruner
from torch.nn import BatchNorm2d
from torch.utils.tensorboard import SummaryWriter

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


# schema_path = "/root/home/code/YOLOX/exps/default/yolox_s_schema.json"
schema_path = "./exps/default/yolox_s_schema.json"
ckpt = "/root/home/code/YOLOX/YOLOX_outputs/yolox_s_s0.0001/best_ckpt.pth"
file_name = "/root/home/code/YOLOX/YOLOX_outputs/yolox_s_bn_offical"


model = get_exp(None, "yolox-s").get_model().eval()
model.apply(random_init)

# ckpt = torch.load(ckpt, map_location="cpu")
# load the model state dict
# model.load_state_dict(ckpt["model"])

pruner = SlimPruner(model, schema_path)
pruning_result = pruner.run(0.65)
x = torch.Tensor(1, 3, 224, 224)
pruned_model = pruner.pruned_model.eval()
pruned_model(x)

# tblogger = SummaryWriter(file_name)
# for name, m in model.named_modules():
#     if isinstance(m, BatchNorm2d):
#         tblogger.add_histogram(f"BN_weights/{name}", m.weight.data.cpu().numpy(), 300)
