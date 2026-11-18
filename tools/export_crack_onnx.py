# -*- coding: utf-8 -*-
"""把下载的 PyTorch 裂缝模型导出成 ONNX,供 crack_app 直接推理。

只做模型转换,不参与网页运行时。需要装 torch / transformers 的环境
(basic_env 里没有 torch,本仓库一般在 clip128 / img_cls 这类环境里跑)::

    C:\\Users\\FANLS\\.conda\\envs\\clip128\\python.exe tools\\export_crack_onnx.py

输出约定(和已有 tom/dihao 一致):输入 ``1x3xHxW``,ImageNet 归一化;
输出 ``1x1xHxW`` 的 **logits**(未过 sigmoid),网页端统一 sigmoid。

支持:
- thaki     ThakiCloud/Thaki-FacilityVision-CrackSeg-E2(UNet-ResNet34,CC-BY-4.0)
- segformer onebeans/segformer_crack_detection(SegFormer-b0,MIT;取 crack 类通道)
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(REPO_ROOT, 'models', 'crack')


def export_thaki(tile=768):
    import torch
    from safetensors.torch import load_file

    mdir = os.path.join(MODELS, 'thaki')
    sys.path.insert(0, mdir)
    from crackseg import unet_resnet34

    net = unet_resnet34(1)
    net.load_state_dict(load_file(os.path.join(mdir, 'model.safetensors')))
    net.eval()
    dummy = torch.zeros(1, 3, tile, tile)
    out = os.path.join(MODELS, 'thaki', 'unet_r34_crackseg_e2.onnx')
    torch.onnx.export(
        net, dummy, out, input_names=['input'], output_names=['logits'],
        opset_version=17, do_constant_folding=True, dynamo=False)
    return out


def export_segformer(mdir, out, tile=512):
    import torch
    import torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        mdir, attn_implementation='eager').eval()
    labels = model.config.id2label
    crack_idx = next((int(k) for k, v in labels.items()
                      if 'crack' in str(v).lower()), 1)
    print('id2label', labels, '-> crack channel', crack_idx)

    class Wrap(torch.nn.Module):
        def __init__(self, m, idx):
            super().__init__()
            self.m = m
            self.idx = idx

        def forward(self, x):
            logits = self.m(pixel_values=x).logits
            logits = F.interpolate(logits, size=x.shape[-2:],
                                   mode='bilinear', align_corners=False)
            return logits[:, self.idx:self.idx + 1]

    dummy = torch.zeros(1, 3, tile, tile)
    torch.onnx.export(
        Wrap(model, crack_idx).eval(), dummy, out,
        input_names=['input'], output_names=['logits'],
        opset_version=17, do_constant_folding=True, dynamo=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='导出裂缝模型到 ONNX')
    ap.add_argument('model', choices=['thaki', 'segformer', 'varcoder'])
    ap.add_argument('--tile', type=int, default=None,
                    help='导出用的固定输入边长(默认 thaki 768 / SegFormer 512)')
    args = ap.parse_args(argv)
    if args.model == 'thaki':
        path = export_thaki(args.tile or 768)
    elif args.model == 'varcoder':
        path = export_segformer(
            os.path.join(MODELS, 'varcoder'),
            os.path.join(MODELS, 'varcoder', 'segformer_b0_crack.onnx'),
            args.tile or 512)
    else:
        path = export_segformer(
            os.path.join(MODELS, 'segformer_b0'),
            os.path.join(MODELS, 'segformer_b0', 'segformer_b0_crack.onnx'),
            args.tile or 512)
    print('saved', path, os.path.getsize(path) / 1e6, 'MB')


if __name__ == '__main__':
    main()
