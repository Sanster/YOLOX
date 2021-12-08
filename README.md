# YOLOX-Slim
Implement [Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) on YOLOX

- [x] Sparse-training -> pruning -> fine-tuning
- [x] Automatic applying network slimming on model without modifying the model code(backbone/fpn/head)
- [x] Export pruned model as onnx model
- [ ] Find good training recipe on COCO:
    - sparsity train epoch/lr
    - hyper parameter of sparsity train: `s`
    - scheduler of `s`: In the original paper the author used a fixed `s` parameter, perhaps using a scheduler could get better results
    - fine-tuning epoch/lr/ratio
- [ ] Make train/fine-tuning with fp16 work


## Experiment
Using YOLOX-s model for sparse training, pruning and then fine-tuning, 
I expected to be able to train a slimming model with better mAP and smaller parameters than YOLOX-Tiny.
However, due to limited GPU resources, both models were not fully trained (120 and 25 rounds), 
and the training hyper parameters and strategies are yet to be explored.

| Model | size | mAP<sup>val<br>0.5:0.95 |  Params<br>(M) | FLOPs<br>(G) | weights | notes |
|-------|------|-------------------------|----------------|--------------|---------|-------|
|YOLOX-s(sparse-training) |640  | 37.9 | 9.0 | 26.8 | [github](https://github.com/Sanster/models/raw/master/YOLOX/yolox_s_slim_sparsity_train/latest_ckpt.pth) | max_epoch 120, linear warm up to `s=0.0001`|
|YOLOX-s(slimming model) |640  |16.5  | 1.58  | 6.43 | [github](https://github.com/Sanster/models/raw/master/YOLOX/yolox_s_slim_fine_tuning/latest_ckpt.pth) [github-onnx](https://github.com/Sanster/models/raw/master/YOLOX/yolox_s_slim_fine_tuning/latest_ckpt.onnx)| network_slim_ratio=0.65, max_epoch=25 |
|YOLOX-s(sparse-training) |640  | 33.8 | 9.0 | 26.8 | [github](https://github.com/Sanster/models/blob/master/YOLOX/yolox_s0.0002_warmup_10/latest_ckpt_s0.0002_warmup_10.pth) | max_epoch 80, linear warm up 10 epoch `s=0.0002`|
|YOLOX-s(slimming model) |640  |26.74  |  |  | [github](https://github.com/Sanster/models/blob/master/YOLOX/yolox_s0.0002_warmup_10/latest_ckpt_s0.0002_warmup_10_fine_tuning_0.6.pth)| network_slim_ratio=0.6, max_epoch=80 |
|YOLOX-s(Official)    |640  |40.5 | 9.0 | 26.8 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
|YOLOX-Tiny(Official) |416  |32.8 | 5.06 | 6.45 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth) |


## Installation

Step1. Setup [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX#quick-start)

Step2. Install network slimming library:

```bash
pip3 install git+https://github.com/Sanster/pytorch-network-slimming.git@0.2.0
```

## Quick Start

Generate pruning schema:
```bash
python3 tools/gen_pruning_schema.py --save-path ./exps/network_slim/yolox_s_schema.json --name yolox-s 
```

Sparse-training, `network_slim_sparsity_train_s` is a hyper parameter that needs to be adjusted according to your data
```bash
python3 tools/train.py -d 4 -b 64 \
-f exps/network_slim/yolox_s_slim_train.py \
-expn yolox_s_slim_sparsity_train \
network_slim_sparsity_train_s 0.0001
```

Apply network slimming and fine-tuning pruned model
```bash
python3 tools/train.py -d 4 -b 64 \
-f exps/network_slim/yolox_s_slim.py \
-c ./YOLOX_outputs/yolox_s_slim_sparsity_train/latest_ckpt.pth \
-expn yolox_s_slim_fine_tuning \
network_slim_ratio 0.65
```

Use slimming model nun `demo.py`
```bash
python3 tools/demo.py image \
-f exps/network_slim/yolox_s_slim.py \
-c ./YOLOX_outputs/yolox_s_slim_fine_tuning/latest_ckpt.pth \
--path assets/dog.jpg  --save_result
```

Use slimming model run `eval.py`
```bash
python3 tools/eval.py -d 4 -b 64 \
-f exps/network_slim/yolox_s_slim.py \
-c ./YOLOX_outputs/yolox_s_slim_fine_tuning/latest_ckpt.pth \
--conf 0.001
```

## Deployment
Export pruned model as onnx model
```bash
python3 tools/export_onnx.py \
-f exps/network_slim/yolox_s_slim.py \
-c ./YOLOX_outputs/yolox_s_slim_fine_tuning/latest_ckpt.pth \
--output-name ./YOLOX_outputs/yolox_s_slim_fine_tuning/latest_ckpt.onnx
```

Run `onnx_inference.py`
```bash
python3 demo/ONNXRuntime/onnx_inference.py \
-m ./YOLOX_outputs/yolox_s_slim_fine_tuning/latest_ckpt.onnx \
-i assets/dog.jpg -o YOLOX_outputs
```
