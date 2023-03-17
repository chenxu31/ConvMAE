# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pdb
import torch
import time
import numpy
import platform
import skimage.io
import glob
import vision_transformer
import models_convmae
from functools import partial
import torch.nn as nn


if platform.system() == 'Windows':
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    UTIL_DIR = r"/home/chenxu/Nutstore Files/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_pelvic_pt as common_pelvic
import common_net_pt as common_net


class ConvViT(vision_transformer.ConvViT):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, clamp_out=False, **kwargs):
        super(ConvViT, self).__init__(**kwargs)
        self.clamp_out = clamp_out

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        x = x + self.pos_embed
        for blk in self.blocks3:
            x = blk(x)

        x = self.norm(x)
        print(222, x.shape)

        outcome = x[:, 0]
        return outcome


def convvit_small_patch16(**kwargs):
    model = ConvViT(
        img_size=[256, 64, 32], patch_size=[4, 2, 2], embed_dim=[128, 256, 384], depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    """
    img_size = [256, 64, 32], patch_size = [4, 2, 2], embed_dim = [128, 256, 384], depth = [2, 2, 11], num_heads = 12,
    decoder_embed_dim = 256, decoder_depth = 4, decoder_num_heads = 8,
    mlp_ratio = [4, 4, 4], norm_layer = partial(nn.LayerNorm, eps=1e-6), ** kwargs)
    """

    return model


def main(device, args):
    #model = models_convmae.__dict__[args.model](in_chans=args.n_slices, clamp_out=True)
    model = convvit_small_patch16(in_chans=args.n_slices, clamp_out=True)
    checkpoint = torch.load(args.checkpoint_file)

    model.load_state_dict(checkpoint["model"], strict=False)

    model.to(device)
    model.eval()

    if args.modality == "ct":
        test_data, _, _, _ = common_pelvic.load_test_data(args.data_dir)
    elif args.modality == "cbct":
        _, test_data, _, _ = common_pelvic.load_test_data(args.data_dir)
    else:
        assert 0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with torch.no_grad():
        in_patch = torch.from_numpy(test_data[0:1, 100:101, :, :]).to(device)

        ret = model(in_patch)
        pdb.set_trace()
        print(loss)
        ret = model.unpatchify(ret)
        ret = ret.cpu().detach().numpy()[0, 0, :, :]
        ret = common_pelvic.data_restore(ret)

        skimage.io.imsave("ori.jpg", common_pelvic.data_restore(in_patch.cpu().detach().numpy()[0, 0, :, :]))#common_pelvic.data_restore(model.unpatchify(model.patchify(in_patch)).cpu().detach().numpy())[0, 0, :, :])
        skimage.io.imsave("syn.jpg", ret)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--model', default='convmae_convvit_small_patch16', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--data_dir', type=str, default=r'/home/chenxu/datasets/pelvic/h5_data_nonrigid', help='path of the dataset')
    parser.add_argument('--checkpoint_file', type=str, default=r'/home/chenxu/training/checkpoints/convnextv2/atto_ct/checkpoint-799.pth', help='path of the dataset')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--decoder_depth', type=int, default=1)
    parser.add_argument('--decoder_embed_dim', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='outputs', help="the output directory")
    parser.add_argument('--n_slices', default=1, type=int, help='number of slices per training sample')
    parser.add_argument('--modality', type=str, default='ct', choices=["ct", "cbct"], help="the output directory")

    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    main(device, args)
