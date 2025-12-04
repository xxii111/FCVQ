import os
os.environ["TORCH_HOME"] = "./dinov2"
import argparse
import random
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
tb_logger = None
from dinov2.hub.backbones import dinov2_vitg14
import dinov2.eval.segmentation.utils.colormaps as colormaps

from model import FCVQ
from functools import partial

from tqdm import tqdm

from PIL import Image

import time

import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose
from mmseg.ops import resize
import types
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)
import torch.nn.functional as F


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

    device = 'cpu'
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



def test_epoch(loss_functioner, fcvq, model, backbone_model, num_embeddings, vq_path):
    device = 'cuda'
    fcvq.eval()
    fcvq.to('cpu')
    model.to(device)
    model.eval()

    hist = np.zeros((21, 21))
    hist_recon = np.zeros((21, 21))
    image_list = []

    # use 'val_20' and VOC2012 to access test dataset
    with open('val_100.txt') as f:
        image_list = f.readlines()
        image_list = ''.join(image_list).strip('\n').splitlines()


    # i=0
    cfg = model.cfg
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    raw_dir = f'./seg/test'
    raw_rec_dir = f'./seg/test'


    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(raw_rec_dir):
        os.makedirs(raw_rec_dir)

  
    mse_list=[]
    mse_list_recon=[]

    fcvq.load_state_dict(torch.load(vq_path)["vqvae_state_dict"])
    
    with torch.no_grad(): 
        bits_total = 0
        bpp_average = 0.0
        num = 0
        enc_time_total = 0.0
        dec_time_total = 0.0
        for image_name in tqdm(image_list):
            num += 1
            image = Image.open(f'./VOC2012/JPEGImages/{image_name}.jpg')
            label = Image.open(f'./VOC2012/SegmentationClass/{image_name}.png')
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
            # aug_feature_dq_list_numpy = np.clip(aug_feature_dq_list_numpy, -10, 10)

            aug_feature_dq_list_numpy_unpack = unpack_dinov2(aug_feature_dq_list_numpy,2,1,1370,1536)
            aug_feature_dq_list_tensor = torch.from_numpy(aug_feature_dq_list_numpy_unpack).to('cpu')
           
            #aug_feature_dq_list_tensor [2, 1370, 1536]
            aug_feature_dq_list_tensor_sq = aug_feature_dq_list_tensor.squeeze(1)
            #==========compress==========
            start_enc = time.time()
            fcvq.uncondi_entropy_model.get_ready_for_compression()
            aug_feature_dq_list_tensor_recon_enc, mse_loss, string, encoding_inds=fcvq.compress(aug_feature_dq_list_tensor_sq)
            end_enc = time.time()
            enc_time_total += (end_enc - start_enc)
            bit_stream_size = len(string)*8
         
            #==========decompress==========
            start_dec = time.time()
            aug_feature_dq_list_tensor_recon_dec =  fcvq.decompress(string, aug_feature_dq_list_tensor_sq.shape)
            end_dec = time.time()
            dec_time_total += (end_dec - start_dec)

            aug_feature_dq_list_tensor_recon = aug_feature_dq_list_tensor_recon_dec.unsqueeze(1)
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

            print(f'Encoded feature: {num} |'
                  f'\tData Size (in bits):{bit_stream_size} |'
                  f'\tData Size (in bpp):{bit_stream_size/(2*1370*1536)}')####################################
        
   
    
            bits_total += bit_stream_size
    all_iou=per_class_iu(hist)
    all_miou=np.nanmean(all_iou)
    all_iou_recon=per_class_iu(hist_recon)
    all_miou_recon=np.nanmean(all_iou_recon)
    bpp_average =  bits_total / ((100)*2740*1536)   ###########################################
    print( f"\t=======Ori-IOU======: ",all_iou  )
    print( f"\t=======Ori-mIOU======: ", all_miou )
    print( f"\t=======Ori-MSE======",np.mean(mse_list))
    print( f"\t=======Recon-IOU======: ",all_iou_recon  )
    print( f"\t=======Recon-mIOU======: ", all_miou_recon )
    print( f"\t=======Recon-MSE======",np.mean(mse_list_recon))
    print("All Data Size (in bits):", bits_total)
    print("Average Data Size (in bpp):", bpp_average)
    return all_miou, all_miou_recon, np.mean(mse_list)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument('--pretrain', type=bool, required=False, default=True, help='fine-tune from a pretrain model')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('-warm_up', default=False, type=bool, help='use step learning rate')
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )

    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size (default: %(default)s)"
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
    parser.add_argument("--seed", type=int,help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, default='/output/', help="Path to a checkpoint")
    parser.add_argument('--embedding_dim', type=int, default=80, help='scale factor')
    parser.add_argument('--num_embeddings', type=int, default=10000, help='scale factor')
    parser.add_argument('--vq_path', type=str, default = './output/')
    parser.add_argument('--num_chunks', type=int, default=1, help='dim of the chunk latent')
    parser.add_argument('--lmbda', type=float, default=5., help='scale factor')
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

    device = 'cpu'
    fcvq=FCVQ(num_embeddings=args.num_embeddings,
                embedding_dim=args.embedding_dim,
                num_chunks=args.num_chunks,
                lmbda=args.lmbda).to(device)

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
 

    loss_functioner = torch.nn.MSELoss()
    
    # all_miou, all_miou_recon, eval_mse = test_epoch(loss_functioner, fcvq, model, backbone_model, args.num_embeddings, args.vq_path)
    test_epoch(loss_functioner, fcvq, model, backbone_model, args.num_embeddings, args.vq_path)

if __name__ == "__main__":
    main(sys.argv[1:])