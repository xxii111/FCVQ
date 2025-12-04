import os
import sys
import json
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from PIL import Image
import torchvision
from dinov2.hub.classifiers import dinov2_vitg14_lc
import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from model import FCVQ
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)
import time


def test_epoch(model, fcvq, vq_path):
    model.eval()
    device = next(model.parameters()).device
    fcvq.eval()
    fcvq.to('cpu')
   
    fcvq.load_state_dict(torch.load(vq_path)["vqvae_state_dict"])
    eval_acc = 0.
    eval_acc_ori = 0.
    eval_mse = 0.
    eval_rate = 0.

    raw_dir='./cls/test'
    with open('./imagenet_selected_label500.txt', "r") as f:
        data = f.readlines()
    ls = [i.split()[0] for i in data]
    lsable=[i.split()[1] for i in data]

    num=0
    enc_time_total = 0.0
    dec_time_total = 0.0
    all_encoding_inds = []
    for x in tqdm(data):
        file_name = x.split()[0]
        y=x.split()[1]
        batch_y = torch.tensor([int(y)]).to(device)
        aug_feature_dq_list_numpy = np.load(f'{raw_dir}/{file_name}.npy')
        aug_feature_dq_list_tensor = torch.from_numpy(aug_feature_dq_list_numpy).to('cpu')
        with torch.no_grad():
            #vq feat
            aug_feature_dq_list_tensor_squeeze = aug_feature_dq_list_tensor.squeeze(0)
            #==========compress==========
            start_enc = time.time()
            fcvq.uncondi_entropy_model.get_ready_for_compression()
            feat_recon_squeeze, mse_loss, string, encoding_inds=fcvq.compress(aug_feature_dq_list_tensor_squeeze)
            end_enc = time.time()
            enc_time_total += (end_enc - start_enc)
            bit_stream_size = len(string)*8
            all_encoding_inds.extend(encoding_inds[0].view(-1).cpu().numpy().tolist())
            #==========decompress==========
            start_dec = time.time()
            feat_recon_squeeze = fcvq.decompress(string, aug_feature_dq_list_tensor_squeeze.shape)
            end_dec = time.time()
            dec_time_total += (end_dec - start_dec)
            import pdb;pdb.set_trace()
            feat_recon = feat_recon_squeeze.unsqueeze(0)
            feat_recon_npy = feat_recon.cpu().numpy()
            #feat_recon:[1,1,257,1536]
            
            aug_feature_dq_list=[[feat_recon[0]]]
            aug_feature_dq_list_ori=[[aug_feature_dq_list_tensor[0]]]
            

            # head
            out_net = model.forward_decode(aug_feature_dq_list[0])
            out_net_ori = model.forward_decode(aug_feature_dq_list_ori[0])
            pred = torch.max(out_net, 1)[1]
            pred_ori = torch.max(out_net_ori, 1)[1]
            # print("pred",pred)
            num_correct = (pred == batch_y).sum()
            num_correct_ori = (pred_ori == batch_y).sum()
            eval_acc += num_correct.item()
            eval_acc_ori += num_correct_ori.item()

            # compute mse
            org_feat = np.load(f'{raw_dir}/{file_name}.npy')
            
            mse = (np.square(org_feat - feat_recon_npy)).mean()
            org_feat_tensor = torch.tensor(org_feat, dtype=torch.float32)
            aug_feature_dq_list_numpy_tensor = torch.tensor(feat_recon_npy, dtype=torch.float32)
            mse_torch = torch.nn.functional.mse_loss(org_feat_tensor, aug_feature_dq_list_numpy_tensor)

            eval_mse += mse
            eval_rate += bit_stream_size
            num += 1
        
    print(num)
    # print(eval_acc_ori)
    eval_mse = eval_mse / (num)
    eval_acc = eval_acc / (num)
    eval_acc_ori = eval_acc_ori / (num)
    eval_rate = eval_rate / ((num)*257*1536)
    ## load compressed feature into head end

    print(f"\t=======MSE: {eval_mse:.6f}=======\n"
        f"\t=======rate: {eval_rate:.6f}=======\n"
        f"\t=====Eval Accuracy: {eval_acc:.6f}=====\n"
        f"\t==Original Feature Eval Accuracy: {eval_acc_ori:.6f}==\n")
    print(f"Average Encoding Time per Image: {enc_time_total / num:.4f} s")
    print(f"Average Decoding Time per Image: {dec_time_total / num:.4f} s")
    print("=============================================\n")
    return


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument('--vq_path', type=str, default = './output/')
    parser.add_argument('--embedding_dim', type=int, default=32, help='scale factor')
    parser.add_argument('--num_embeddings', type=int, default=2048, help='scale factor')
    parser.add_argument('--num_chunks', type=int, default=1, help='scale factor')
    parser.add_argument('--lmbda', type=float, default=1., help='lambda')
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    print("$$$$$$$$$$$$$$$", torch.__version__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    net = dinov2_vitg14_lc(layers=1, pretrained=True)
    net.to(device)
    fcvq=FCVQ(num_embeddings=args.num_embeddings,
                embedding_dim=args.embedding_dim,
                num_chunks=args.num_chunks,
                lmbda=args.lmbda).to(device)
    test_epoch(net, fcvq, args.vq_path)

if __name__ == '__main__':
    main(sys.argv[1:])