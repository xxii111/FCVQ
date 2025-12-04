import os
os.environ["TORCH_HOME"] = "./dinov2"

import argparse
import random
import shutil
import sys
import math
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
tb_logger = None
from collections import OrderedDict
from model import FCVQ
from torch.utils.data import DataLoader, Subset
import torchvision
import re
# from torch.cuda.amp import autocast
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)
from functools import partial
from PIL import Image
from dinov2.hub.backbones import dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14
import dinov2.eval.segmentation.utils.colormaps as colormaps
import dinov2.eval.segmentation.models
import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
from mmseg.ops import resize
import re
from dataset_seg import Dinov2DatasetTrain, Dinov2DatasetTest
# from torch.cuda.amp import autocast
import types
from tqdm import tqdm
import torch.nn.functional as F

# def total_variation_loss(img):
#      bs_img, c_img, h_img, w_img = img.size()
#      tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
#      tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
#      return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)
def encode_decode_decode(self, crop_feature_list, img_metas, backbone_model, shape):
    outputs = [backbone_model.norm(out) for out in crop_feature_list]
    # class_tokens = [out[:, 0] for out in outputs]
    outputs = [out[:, 1 + backbone_model.num_register_tokens:] for out in outputs]
    B = outputs[0].shape[0]
    w, h = shape[0], shape[1]
    outputs = [
        out.reshape(B, math.ceil(w / backbone_model.patch_size), math.ceil(h / backbone_model.patch_size), -1).permute(
            0, 3, 1, 2).contiguous()
        for out in outputs
    ]
    x = tuple(outputs)
    if self.with_neck:
        x = self.neck(x)
    out = self._decode_head_forward_test(x, img_metas)
    out = resize(
        input=out,
        size=shape,
        mode='bilinear',
        align_corners=self.align_corners)
    return out

def slide_inference_encode(self, img, img_meta, rescale):
    """Inference by sliding-window with overlap.

    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.
    """

    h_stride, w_stride = self.test_cfg.stride
    h_crop, w_crop = self.test_cfg.crop_size
    batch_size, _, h_img, w_img = img.size()
    num_classes = self.num_classes
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

    feature_list = []
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            crop_feature_list = self.backbone(crop_img, is_encode=True)
            feature_list.append(crop_feature_list)

    return feature_list

def unpack_dinov2(pack_feat, N, C, H, W):
    unpack_feat = pack_feat.reshape(N, H, C, W).transpose(0, 2, 1, 3)  # unpack (NxH,CxW) to (N,C,H,W) (2,4,1611,1536)
    return unpack_feat

def slide_inference_decode(self, feature_list, img_meta, rescale, backbone_model):
    """Inference by sliding-window with overlap.

    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.
    """
    device = next(backbone_model.parameters()).device

    h_stride, w_stride = self.test_cfg.stride
    h_crop, w_crop = self.test_cfg.crop_size
    batch_size = feature_list[0][0].shape[0]
    h_img, w_img = img_meta[0]['img_shape'][0], img_meta[0]['img_shape'][1]
    num_classes = self.num_classes
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = torch.zeros((batch_size, num_classes, h_img, w_img)).to(device)
    count_mat = torch.zeros((batch_size, 1, h_img, w_img)).to(device)
    i = 0
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_seg_logit = self.encode_decode_decode(feature_list[i], img_meta, backbone_model,
                                                       self.test_cfg.crop_size)
            preds += F.pad(crop_seg_logit,
                           (int(x1), int(preds.shape[3] - x2), int(y1),
                            int(preds.shape[2] - y2)))

            count_mat[:, :, y1:y2, x1:x2] += 1
            i += 1
    assert (count_mat == 0).sum() == 0
    if torch.onnx.is_in_onnx_export():
        # cast count_mat to constant while exporting to ONNX
        count_mat = torch.from_numpy(
            count_mat.cpu().detach().numpy()).to(device=device)
    preds = preds / count_mat
    if rescale:
        # remove padding area
        resize_shape = img_meta[0]['img_shape'][:2]
        preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
        preds = resize(
            preds,
            size=img_meta[0]['ori_shape'][:2],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
    return preds

def simple_test_encode(self, img, img_meta, rescale=True):
    """Simple test with single image."""
    assert self.test_cfg.mode in ['slide', 'whole']
    ori_shape = img_meta[0]['ori_shape']
    assert all(_['ori_shape'] == ori_shape for _ in img_meta)
    if self.test_cfg.mode == 'slide':
        feature_list = self.slide_inference_encode(img, img_meta, rescale)
    else:
        assert False

    return feature_list

def simple_test_decode(self, feature_list, img_meta, backbone_model, rescale=True):
    """Simple test with single image."""
    seg_logit = self.slide_inference_decode(feature_list, img_meta, rescale, backbone_model=backbone_model)
    output = F.softmax(seg_logit, dim=1)
    flip = img_meta[0]['flip']
    if flip:
        flip_direction = img_meta[0]['flip_direction']
        assert flip_direction in ['horizontal', 'vertical']
        if flip_direction == 'horizontal':
            output = output.flip(dims=(3,))
        elif flip_direction == 'vertical':
            output = output.flip(dims=(2,))
    seg_logit = output
    seg_pred = seg_logit.argmax(dim=1)
    if torch.onnx.is_in_onnx_export():
        # our inference backend only support 4D output
        seg_pred = seg_pred.unsqueeze(0)
        return seg_pred
    seg_pred = seg_pred.cpu().numpy()
    # unravel batch dim
    seg_pred = list(seg_pred)
    return seg_pred

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

def create_segmenter(cfg, backbone_model):
    cfg.model.backbone.out_indices = [39]
    cfg.model.decode_head.in_index = [0]
    model = init_segmentor(cfg)
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
    model.init_weights()
    return model

def fast_hist(label, prediction, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + prediction[k].astype(int), minlength=n * n).reshape(n, n)

def per_class_iu(hist):
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return iou

def compute_miou(label, prediction, n):
    hist = fast_hist(label, prediction, n)
    iou = per_class_iu(hist)
    return iou

def compute_pa(label, prediction, n):
    hist = fast_hist(label, prediction, n)
    pa=np.sum(np.diag(hist))/np.sum(hist)
    return pa

class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

def colorful(out, name):
    ## call me to save a seg img
    arr = out.astype(np.uint8)
    im = Image.fromarray(arr)

    palette = []
    for i in range(256):
        palette.extend((i, i, i))
    palette[:3 * 21] = np.array([[0, 0, 0],
                                 [128, 0, 0],
                                 [0, 128, 0],
                                 [128, 128, 0],
                                 [0, 0, 128],
                                 [128, 0, 128],
                                 [0, 128, 128],
                                 [128, 128, 128],
                                 [64, 0, 0],
                                 [192, 0, 0],
                                 [64, 128, 0],
                                 [192, 128, 0],
                                 [64, 0, 128],
                                 [192, 0, 128],
                                 [64, 128, 128],
                                 [192, 128, 128],
                                 [0, 64, 0],
                                 [128, 64, 0],
                                 [0, 192, 0],
                                 [128, 192, 0],
                                 [0, 64, 128]
                                 ], dtype='uint8').flatten()
    im.putpalette(palette)
    im.save(f'/output/results/VOC2012/Segmentation/comp6_test_cls/{name}.png')

def render_segmentation(segmentation_logits, dataset):
    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1]
    return Image.fromarray(segmentation_values)    

def train_one_epoch(fcvq, loss_functioner, train_loader, test_loader, optimizer, epoch, feature_split_size):
    device = 'cuda'
    # 设置VQ为train
    fcvq.train()
    fcvq.to(device)
    for batch_idx, batch in enumerate(train_loader):
        #cls feat shape:[N,257,1536]
        #seg feat shape:[N,1370,1536]
        feat = batch[0].to(device) 
        feat = torch.clamp(feat, min=-5, max=5)
        feature_split_size
        N,H,W = feat.shape
        num_splits = W // feature_split_size
        assert W % feature_split_size == 0, "1536 must be divisible by split size"
        
        recon_chunks = []
        total_rd_loss = 0
        total_mse_loss = 0
        total_rate = 0

        optimizer.zero_grad()

        for i in range(num_splits):
            start = i * feature_split_size
            end = (i + 1) * feature_split_size
            sub_feat = feat[:, :, start:end]
        
            feat_recon, mse_loss,  rd_loss, rate, encoding_inds=fcvq(sub_feat)
            mse_loss = loss_functioner(feat_recon, sub_feat)

        # loss = mse_loss + lambda_d*tv_loss_recon
            loss = rd_loss

            recon_chunks.append(feat_recon.detach())  # detach to avoid keeping graph
            total_rd_loss += rd_loss.item()
            total_mse_loss += mse_loss.item()
            total_rate += rate.item()
        # print(tv_loss)
            loss.backward()

        optimizer.step()

        feat_recon_full = torch.cat(recon_chunks, dim=-1)


        steps = epoch * len(train_loader) + batch_idx

        tb_logger.add_scalar('lr',optimizer.param_groups[0]['lr'], steps)
        tb_logger.add_scalar('Train mse',mse_loss, steps)
        tb_logger.add_scalar('Train rate',rate, steps)
        tb_logger.add_scalar('Train loss',rd_loss, steps)

        print(
            f"\tTrain epoch {epoch}"
            f"\tTrain batch {batch_idx}: ["
            f"\t{batch_idx* train_loader.batch_size}/{len(train_loader.dataset)}"
            f"\t({100. * batch_idx / len(train_loader):.0f}%)]\n"
            f'\tTrain mse loss: {mse_loss.item():.4f} |\n'
            f'\tTrain rate: {rate.item():.4f} |\n'  
            f'\tTrain rd loss: {rd_loss.item():.4f} |'          
        )
        # if i > 1000:
        #     break

def validate_epoch(epoch, test_loader, loss_functioner, fcvq, model, backbone_model, optimizer):
    
    device = 'cuda'
    fcvq.to(device)
    fcvq.eval()

    model.to(device)
    model.eval()

    hist = np.zeros((21, 21))
    hist_recon = np.zeros((21, 21))
    image_list = []
    # myf
    # use '/data/myfriend/SBD/SBD/train.txt' and SBD dataset to save train dataset
    # use 'val_20' and VOC2012 to access test dataset
    with open('val_100.txt') as f:
        image_list = f.readlines()
        image_list = ''.join(image_list).strip('\n').splitlines()

    # i=0
    cfg = model.cfg
    device = next(model.parameters()).device
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    raw_dir = f'/data/qiaoxichen/model/dinov2_dataset/seg/test'
    raw_rec_dir = f'/data/qiaoxichen/model/dinov2_dataset/seg/test'

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(raw_rec_dir):
        os.makedirs(raw_rec_dir)

  
    mse_list=[]
    mse_list_recon=[]
    with torch.no_grad():
        for image_name in tqdm(image_list):
            image = Image.open(f'/data/bitahub/VOC2012/JPEGImages/{image_name}.jpg')
            label = Image.open(f'/data/bitahub/VOC2012/SegmentationClass/{image_name}.png')
            img = np.array(image)[:, :, ::-1]  # BGR

            # prepare data
            data = dict(img=img)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            if next(model.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [device])[0]
            else:
                data['img_metas'] = [i.data[0] for i in data['img_metas']]


            org_feat = np.load(f'{raw_dir}/{image_name}.npy')
           
            aug_feature_dq_list_numpy = np.load(f'{raw_rec_dir}/{image_name}.npy') # (2740,1536)
            aug_feature_dq_list_numpy = np.clip(aug_feature_dq_list_numpy, -5, 5)

            aug_feature_dq_list_numpy_unpack = unpack_dinov2(aug_feature_dq_list_numpy,2,1,1370,1536)
            aug_feature_dq_list_tensor = torch.from_numpy(aug_feature_dq_list_numpy_unpack).to(device)
           
            #aug_feature_dq_list_tensor [2, 1370, 1536]
            aug_feature_dq_list_tensor_sq = aug_feature_dq_list_tensor.squeeze(1)
            aug_feature_dq_list_tensor_recon_sq, mse_loss, rd_lsos, rate, encoding_inds=fcvq(aug_feature_dq_list_tensor_sq)
            
            aug_feature_dq_list_tensor_recon = aug_feature_dq_list_tensor_recon_sq.unsqueeze(1)
            #original segmentation
            aug_feature_dq_list = []
            for i in range(aug_feature_dq_list_tensor.shape[0]):
                feature_dq_list = []
                for j in range(aug_feature_dq_list_tensor.shape[1]):
                    feature_dq_list.append(aug_feature_dq_list_tensor[i][j].unsqueeze(0))
                aug_feature_dq_list.append(feature_dq_list)
            
            pred = model.simple_test_decode(aug_feature_dq_list, data['img_metas'][0], backbone_model, rescale=True)

            # colorful(pred[0], image_name)   # save a seg img

            array_label = np.array(label)
            hist+=fast_hist(array_label,pred[0],21)

            mse = (np.square(org_feat - aug_feature_dq_list_numpy)).mean()
            mse_list.append(mse)
        #==========================================================
            # recon segmentation
            aug_feature_dq_list_recon = []
            for i in range(aug_feature_dq_list_tensor_recon.shape[0]):
                feature_dq_list_recon = []
                for j in range(aug_feature_dq_list_tensor_recon.shape[1]):
                    feature_dq_list_recon.append(aug_feature_dq_list_tensor_recon[i][j].unsqueeze(0))
                aug_feature_dq_list_recon.append(feature_dq_list_recon)
            
            pred_recon = model.simple_test_decode(aug_feature_dq_list_recon, data['img_metas'][0], backbone_model, rescale=True)

        # colorful(pred[0], image_name)   # save a seg img

            array_label = np.array(label)
            hist_recon+=fast_hist(array_label,pred_recon[0],21)
         
            recon_feat = aug_feature_dq_list_tensor_recon.cpu().numpy()
            mse_recon = (np.square(org_feat - recon_feat)).mean()
            mse_list_recon.append(mse_recon)

    all_iou=per_class_iu(hist)
    all_miou=np.nanmean(all_iou)
    all_iou_recon=per_class_iu(hist_recon)
    all_miou_recon=np.nanmean(all_iou_recon)
    print( f"\t=======Ori-IOU======: ",all_iou  )
    print( f"\t=======Ori-mIOU======: ", all_miou )
    print( f"\t=======Ori-MSE======",np.mean(mse_list))
    print( f"\t=======Recon-IOU======: ",all_iou_recon  )
    print( f"\t=======Recon-mIOU======: ", all_miou_recon )
    print( f"\t=======Recon-rate======: ", rate)
    print( f"\t=======Recon-MSE======",np.mean(mse_list_recon))
    ## load compressed feature into head end
    step = epoch

    # tb_logger.add_scalar('IOU', all_iou, step)
    tb_logger.add_scalar('Ori-mIOU', all_miou, step)
    tb_logger.add_scalar('Ori-MSE', np.mean(mse_list), step)
    tb_logger.add_scalar('Recon-mIOU', all_miou_recon, step)
    tb_logger.add_scalar('Recon-MSE', np.mean(mse_list_recon), step)
    tb_logger.add_scalar('Recon-rate', rate, step)
    return all_miou, all_miou_recon, np.mean(mse_list_recon)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-e",
        "--epochs",
        default=1,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )

    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, default=3407, help="Set random seed for reproducibility")
    parser.add_argument("--checkpoint", type=str, default='/output/', help="Path to a checkpoint")
    parser.add_argument('--scale_init', type=float, default=10., help='scale factor')
    parser.add_argument('--embedding_dim', type=int, default=10, help='scale factor')
    parser.add_argument('--num_embeddings', type=int, default=32, help='scale factor')
    parser.add_argument('--num_chunks', type=int, default=1, help='dim of the chunk latent')
    parser.add_argument('--pretrained_path', type=str, default='/output/', help='dim of the chunk latent')
    parser.add_argument("--pretrained", type=bool, default=False, help="load pretrained codebook")
    parser.add_argument('--lmbda', type=float, default=1., help='scale factor')
    parser.add_argument('--feature_split_size', type=int, default=8, help='dim of the chunk latent')
    
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    global tb_logger
    tb_logger = SummaryWriter(os.path.join('/output/', 'events'))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    
    device = 'cuda'
    
    
    BACKBONE_SIZE = "giant"  # in ("small", "base", "large" or "giant")

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = dinov2_vitg14(pretrained=True)

    backbone_model.eval()
    backbone_model.cuda()

    HEAD_SCALE_COUNT = 3  # more scales: slower but better results, in (1,2,3,4,5)
    HEAD_DATASET = "voc2012"  # in ("ade20k", "voc2012")
    HEAD_TYPE = "linear"  # in ("ms, "linear")

    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

    # cfg_str = load_config_from_url(head_config_url)
    # cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
    cfg = mmcv.Config.fromfile(f"cfg/dinov2_vitg14_{HEAD_DATASET}_{HEAD_TYPE}_config.py")

    if HEAD_TYPE == "ms":
        cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
        print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

    model = create_segmenter(cfg, backbone_model=backbone_model)

    model.simple_test_encode = types.MethodType(simple_test_encode, model)
    model.simple_test_decode = types.MethodType(simple_test_decode, model)
    model.slide_inference_encode = types.MethodType(slide_inference_encode, model)
    model.slide_inference_decode = types.MethodType(slide_inference_decode, model)
    model.encode_decode_decode = types.MethodType(encode_decode_decode, model)

    #myf, head_checkpoint_url can be replaced to your path to weights directly
    load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    model.cuda()
    model.eval()

    DATASET_COLORMAPS = {
        "ade20k": colormaps.ADE20K_COLORMAP,
        "voc2012": colormaps.VOC2012_COLORMAP,
    }

    
    
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    
    fcvq=FCVQ(num_embeddings=args.num_embeddings,
                embedding_dim=args.embedding_dim,
                num_chunks=args.num_chunks,
                lmbda=args.lmbda).to(device)

    if args.pretrained is not None:
    
        # fcvq.load_state_dict(torch.load(args.pretrained_path)["state_dict"])
        print('==========load pretrained codebook successfully=========')

    test_dataset = Dinov2DatasetTest(r"./ILSVRC2012_tiny/val",
                                        transform=data_transform["test"])
    train_dataset = Dinov2DatasetTrain(train=True)

    num_workers = min([os.cpu_count(), args.test_batch_size if args.test_batch_size > 1 else 0, 4])  # number of workers
    print('Using {} dataloader workers every process'.format(num_workers))    

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.test_batch_size, shuffle=False,
                                                  num_workers=num_workers)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=args.batch_size, shuffle=True,
                                                  num_workers=4)
    print('total training images: {}; total training batches: {}'.format(len(train_dataset), len(train_loader)))
    print('total testing images: {}; total testing batches: {}'.format(len(test_dataset), len(test_loader)))

    optimizer = torch.optim.Adam(fcvq.parameters(), lr=args.lr)
    step_size = 20 
    gamma = 0.9 
    train_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_functioner = torch.nn.MSELoss()

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)


    for epoch in range(1, args.epochs+1):
        if optimizer.param_groups[0]['lr'] < 1e-6:
            optimizer.param_groups[0]['lr'] = 1e-6
            print('=======set min lr to 1e-6========')
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        all_miou, all_miou_recon, eval_mse = validate_epoch(epoch, test_loader, loss_functioner, fcvq, model,backbone_model, optimizer) 
        
        train_one_epoch(fcvq, loss_functioner, train_loader, test_loader, optimizer, epoch, args.feature_split_size)
        train_scheduler.step()  
        if args.save:
            if epoch == args.epochs:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "lr": optimizer.param_groups[0]['lr'],
                        "vqvae_state_dict": fcvq.state_dict(),
                        "mIOU": all_miou_recon,
                        "optimizer": optimizer.state_dict(),
                    },
                    epoch == args.epochs,
                    filename=args.checkpoint + 'epoch_'+ str(args.epochs) + 'num_'+ str(args.num_embeddings) + 'chunk_'+ str(args.num_chunks) +'.pth.tar'
                )
                print(f'\ttest loss:{eval_mse:.4f}|'
                    f'\tTest mIOU:{all_miou_recon:.4f}|'
                    f'\tTest mIOU ORI:{all_miou:.4f}|'
                    f'\tsave last epoch model of epoch: {epoch}')


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn", force=True)
    main(sys.argv[1:])