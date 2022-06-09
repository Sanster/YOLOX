#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn
import torch

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from ..utils import xyxy2cxcywh


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x)
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)
            prediction = outputs
            box_corner = prediction.new(prediction.shape)
            # print("first", prediction[:, :, :4])
            box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2) / 416
            box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2) / 416
            box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2) / 416
            box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2) / 416
            prediction[:, :, :4] = box_corner[:, :, :4]
            num_classes = 3

            conf_thre = 0.6
            image_pred = prediction.squeeze(0)

            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
            # MAX確信度の値 class_conf, そのindex class_pred
            value, indexes = torch.topk(class_conf, k=20, dim=0)
            boxes, test, confidences = torch.split(image_pred, (4, 1, 3), dim=1)

            conf_mask = (test[indexes] >= conf_thre).squeeze()

            indexes = indexes[conf_mask]
            cpu_device = torch.device("cpu")
            boxes = boxes.to(cpu_device)
            confidences = confidences.to(cpu_device)
            sorted_confidences = torch.gather(confidences, 0, indexes.expand(-1, 3))
            sorted_boxes = torch.gather(boxes, 0, indexes.expand(-1, 4))

            center_box = xyxy2cxcywh(sorted_boxes)
            detections = {
                "pred_logits": sorted_confidences.unsqueeze(0),
                "pred_boxes": center_box.unsqueeze(0),
            }
        return detections
