import os
import argparse
import random
import sys
import torch
import numpy as np
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
tb_logger = None
from model import FCVQ
import torchvision.transforms as transforms
# from torch.cuda.amp import autocast
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)
from dinov2.hub.classifiers import  dinov2_vitg14_lc
from dataset_cls import Dinov2DatasetTrain
from tqdm import tqdm

def train_one_epoch(fcvq, loss_functioner, train_loader, optimizer, epoch):

    device = 'cuda'

    fcvq.train()
    fcvq.to(device)
    i = 0
    for batch_idx, batch in enumerate(train_loader):
    
        
        feat = batch[0].to(device) #[N,257,1536]
        feat = torch.clamp(feat, min=-10, max=10)
        optimizer.zero_grad()
        # print('train feat shape:', feat.shape)
        # feat_slice = feat[:, :256, :]
        feat_recon, mse_loss, rd_loss, rate, encoding_inds=fcvq(feat)
        mse_loss = loss_functioner(feat_recon, feat)
        loss = rd_loss
        loss.backward()
        optimizer.step()
    
        i = i+1
        steps = epoch * len(train_loader) + batch_idx
        if steps % 1 == 0:
            tb_logger.add_scalar('lr',optimizer.param_groups[0]['lr'], steps)
            tb_logger.add_scalar('Train mse',mse_loss, steps)
            tb_logger.add_scalar('Train rate',rate, steps)
            tb_logger.add_scalar('Train loss',loss, steps)
                
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

def validate_epoch(epoch, loss_functioner, fcvq, dino_net, optimizer):
    
    device = 'cuda'
    fcvq.to(device)
    fcvq.eval()

    dino_net.to(device)
    dino_net.eval()

    eval_acc = 0.
    eval_acc_ori = 0.
    eval_mse=0.
    eval_rate=0.
    raw_dir='./cls/test'

    with open('./imagenet_selected_label500.txt', "r") as f:
        data = f.readlines()
    ls = [i.split()[0] for i in data]
    lsable=[i.split()[1] for i in data]

    with torch.no_grad():
        num = 0
        for x in tqdm(data):
            file_name = x.split()[0]
            y=x.split()[1]
            batch_y = torch.tensor([int(y)]).to(device)

            aug_feature_dq_list_numpy = np.load(f'{raw_dir}/{file_name}.npy')           

            aug_feature_dq_list_tensor = torch.from_numpy(aug_feature_dq_list_numpy).to(device)

            
            aug_feature_dq_list_tensor_squeeze = aug_feature_dq_list_tensor.squeeze(0)#[257,1536]
       
            feat_recon_squeeze, mse_loss, rd_loss, rate, encoding_inds=fcvq(aug_feature_dq_list_tensor_squeeze)
            
            feat_recon = feat_recon_squeeze.unsqueeze(0)
            feat_recon_npy = feat_recon.cpu().numpy()
            
            aug_feature_dq_list=[[feat_recon[0]]]
            aug_feature_dq_list_ori=[[aug_feature_dq_list_tensor[0]]]
            # head
            out_net = dino_net.forward_decode(aug_feature_dq_list[0])
            out_net_ori = dino_net.forward_decode(aug_feature_dq_list_ori[0])

            pred = torch.max(out_net, 1)[1]
            pred_ori = torch.max(out_net_ori, 1)[1]
            
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
            eval_rate += rate
            num += 1
            print(num)
    eval_mse = eval_mse / (num)
    eval_acc = eval_acc / (num)
    eval_acc_ori = eval_acc_ori / (num)
    eval_rate = eval_rate / (num)
    ## load compressed feature into head end
    step = epoch
    tb_logger.add_scalar('eval_mse', eval_mse, step)
    tb_logger.add_scalar('eval_acc', eval_acc, step)
    tb_logger.add_scalar('eval_acc_ori', eval_acc_ori, step)
    tb_logger.add_scalar('eval_rate', eval_rate, step)
    
    print('len test dataset: ', num)
    print(f"\t=======MSE: {eval_mse:.6f}=======\n"
        f"\t=======rate: {eval_rate:.6f}=======\n"
        f"\t=====Eval Accuracy: {eval_acc:.6f}=====\n"
        f"\t==Original Feature Eval Accuracy: {eval_acc_ori:.6f}==\n")

    return eval_mse, eval_acc, eval_acc_ori


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size (default: %(default)s)"
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
    parser.add_argument('--embedding_dim', type=int, default=32, help='scale factor')
    parser.add_argument('--num_embeddings', type=int, default=2, help='scale factor')
    parser.add_argument('--num_chunks', type=int, default=1, help='scale factor')
    parser.add_argument('--lmbda', type=float, default=1., help='lambda')
    parser.add_argument(
        "--pretrained", type=bool, default=False, help="load pretrained model")
    parser.add_argument( "--pretrained_path", type=str, default='/output/', help="pretrained model path")
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
    
    dino_net = dinov2_vitg14_lc(layers=1, pretrained=True)
    dino_net.to(device)
    
    fcvq=FCVQ(num_embeddings=args.num_embeddings,
                embedding_dim=args.embedding_dim,
                num_chunks=args.num_chunks,
                lmbda=args.lmbda).to(device)
    

    train_dataset = Dinov2DatasetTrain(train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=args.batch_size, shuffle=True,
                                                  num_workers=4)
    print('total training images: {}; total training batches: {}'.format(len(train_dataset), len(train_loader)))

    # steplr
    # if args.pretrained:
    #     print(args.pretrained)
    #     fcvq.load_state_dict(torch.load(args.pretrained_path)["vqvae_state_dict"])
    #     lr = torch.load(args.pretrained_path)["lr"]
    # else:
    #     lr = args.lr
    lr = args.lr
    optimizer = torch.optim.Adam(fcvq.parameters(), lr=lr)
    
    step_size = 10 
    gamma = 0.5 
    train_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_functioner = torch.nn.MSELoss()

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)


    for epoch in range(1, args.epochs+1):
        if optimizer.param_groups[0]['lr'] < 1e-6:
            optimizer.param_groups[0]['lr'] = 1e-6
            print('=======set min lr to 1e-6========')
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        eval_mse, eval_acc, eval_acc_ori = validate_epoch(epoch, loss_functioner, fcvq, dino_net, optimizer) 

        train_one_epoch(fcvq, loss_functioner, train_loader, optimizer, epoch)

        train_scheduler.step()
        if args.save:
            if epoch == args.epochs:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "lr": optimizer.param_groups[0]['lr'],
                        "vqvae_state_dict": fcvq.state_dict(),
                        "acc": eval_acc,
                        "optimizer": optimizer.state_dict(),
                    },
                    epoch == args.epochs,
                    filename=args.checkpoint + 'epoch_'+ str(args.epochs) + 'num_'+ str(args.num_embeddings) + 'chunk_'+ str(args.num_chunks) +'.pth.tar'
                )
                print(f'\tTest ACC:{eval_acc:.4f}|'
                    f'\tTest ACC ORI:{eval_acc_ori:.4f}|'
                    f'\tsave last epoch model of epoch: {epoch}')


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn", force=True)
    main(sys.argv[1:])