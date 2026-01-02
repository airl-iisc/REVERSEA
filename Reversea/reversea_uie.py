import os
import argparse
import numpy as np
import torch
from time import time
from torchvision.utils import save_image
from PIL import Image
import gzip
import torch
import pandas as pd
from torch.utils.data import DataLoader
from data_loader import PairedRGBDepthDataset
from scores import getUCIQE,getUIQM
from reversea_model import ReverseaNet,ReverseaLoss
try:
    from tqdm import trange
except:
    trange = range

def main(args):
    uciqe_values = []
    uqims_values = []
    uicm_values = []
    uism_values = []
    uicomn_values = []
    output_names = []
    seed = int(torch.randint(9223372036854775807, (1,))[0]) if args.seed is None else args.seed
    if args.seed is None:
        print('Seed:', seed)
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    train_dataset = PairedRGBDepthDataset(args.images, args.depth, args.depth_16u, args.mask_max_depth, args.height,
                                             args.width, args.device)
    save_dir = args.output
    os.makedirs(save_dir, exist_ok=True)
    check_dir = args.checkpoints
    os.makedirs(check_dir, exist_ok=True)
    target_batch_size = args.batch_size
    dataloader = DataLoader(train_dataset, batch_size=target_batch_size, shuffle=False)
    rs_model = ReverseaNet().to(device=args.device)
    rs_criterion = ReverseaLoss().to(device=args.device)
    rs_optimizer = torch.optim.Adam(rs_model.parameters(), lr=args.init_lr)
    skip_right = True
    total_rs_eval_time = 0.
    total_rs_evals = 0
    total_at_eval_time = 0.
    total_at_evals = 0
    for j, (left, depth, fnames) in enumerate(dataloader):
        print("training")
        image_batch = left
        batch_size = image_batch.shape[0]
        for iter in trange(args.init_iters if j == 0 else args.iters):  # Run first batch for 500 iters, rest for 50
            start = time()
            A1,A,J,direct,div= rs_model(image_batch, depth)
            rs_loss = rs_criterion(A,A1,J)
            rs_optimizer.zero_grad()
            rs_loss.backward()
            rs_optimizer.step()
            total_rs_eval_time += time() - start
            total_rs_evals += batch_size
        direct_mean = direct.mean(dim=[2, 3], keepdim=True)
        direct_std = direct.std(dim=[2, 3], keepdim=True)
        direct_z = (direct - direct_mean) / direct_std
        clamped_z = torch.clamp(direct_z, -5, 5)
        direct_no_grad = torch.clamp(
            (clamped_z * direct_std) + torch.maximum(direct_mean, torch.Tensor([1. / 255]).to(device=args.device)), 0, 1).detach()
        for iter in trange(args.init_iters if j == 0 else args.iters):  # Run first batch for 500 iters, rest for 50
            start = time()
            total_at_eval_time += time() - start
            total_at_evals += batch_size
        print("Losses: %.9f" % (rs_loss.item()))
        avg_rs_time = total_rs_eval_time / total_rs_evals * 1000
        avg_at_time = total_at_eval_time / total_at_evals * 1000
        avg_time = avg_rs_time + avg_at_time
        print("Avg time per eval: %f ms (%f ms bs, %f ms at)" % (avg_time, avg_rs_time, avg_at_time))
        img = image_batch.cpu()
        direct_img = torch.clamp(direct_no_grad, 0., 1.).cpu()
        A1_img = torch.clamp(A1, 0., 1.).cpu()
        A_img = torch.clamp(A, 0., 1.).detach().cpu()
        J_img = torch.clamp(J, 0., 1.).cpu()
        div_img = torch.clamp(div, 0., 1.).cpu()

 
        for side in range(1 if skip_right else 2):
            side_name = 'left' if side == 0 else 'right'
            names = fnames[side]
            for n in range(batch_size):
                i = n + target_batch_size * side
                #if args.save_intermediates:
                    #save_image(direct_img[i], "%s/%s-direct.png" % (save_dir, names[n].rstrip('.png')))
                    #save_image(A[i], "%s/%s-A_img.png" % (save_dir, names[n].rstrip('.png')))
                    # save_image(backscatter_img[i], "%s/%s-A.png" % (save_dir, names[n].rstrip('.png')))
                save_image(J_img[i], "%s/%s-corrected.png" % (save_dir, names[n].rstrip('.png')))
                output_image_path ="%s/%s-corrected.png" % (save_dir, names[n].rstrip('.png'))
                output_image =Image.open(output_image_path)
                output_image = output_image.resize((256, 256))
                image = output_image.convert('RGB')
                image_array = np.array(image)
                uciqe_value = getUCIQE(output_image_path)
                print('UCIQE:',uciqe_value)
                uqims_value,uicm_value,uism_value,uicomn_value =getUIQM(image_array)
                print('UQIMS:',uqims_value)
                uciqe_values.append(uciqe_value)
                uqims_values.append(uqims_value)
                uicm_values.append(uicm_value)
                uism_values.append(uism_value)
                uicomn_values.append(uicomn_value)
                output_names.append(names[n])

        # Save checkpoint with compression
        checkpoint_path = os.path.join(check_dir, f'model_checkpoint_{j}.pth')
        with gzip.open(checkpoint_path, 'wb') as f:
            torch.save({
            'rs_model_state_dict': rs_model.state_dict(),
            'rs_optimizer_state_dict': rs_optimizer.state_dict(),
            }, f)
            
    df = pd.DataFrame({
        'Output Image Name': output_names,
        'uciqe': uciqe_values,
        'uqims': uqims_values,
        'uicm':uicm_values,
        'uism':uism_values,
        'uicomn':uicomn_values
        })
    
    excel_path = os.path.join(save_dir, 'evaluation_metrics.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Evaluation metrics saved to {excel_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--images', type=str, required=True, help='Path to the images folder')
    parser.add_argument('--depth', type=str, required=True , help='Path to the depth folder')
    parser.add_argument('--output', type=str,required=True,  help='Path to the output folder')
    parser.add_argument('--checkpoints', type = str, required=True,help='Path to the checkpoint folder')
    parser.add_argument('--height', type=int, default=720, help='Height of the image and depth files')
    parser.add_argument('--width', type=int, default=1280, help='Width of the image and depth')
    parser.add_argument('--depth_16u', action='store_true',
                        help='True if depth images are 16-bit unsigned (millimetres), false if floating point (metres)')
    parser.add_argument('--mask_max_depth', action='store_true',
                        help='If true will replace zeroes in depth files with max depth')
    parser.add_argument('--seed', type=int, default=None, help='Seed to initialize network weights (use 1337 to replicate paper results)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing images')
    parser.add_argument('--save_intermediates', action='store_true', default=False, help='Set to True to save intermediate files (A, attenuation, and direct images)')
    parser.add_argument('--init_iters', type=int, default=500, help='How many iterations to refine the first image batch (should be >= iters)')
    parser.add_argument('--iters', type=int, default=150, help='How many iterations to refine each image batch')
    parser.add_argument('--init_lr', type=float, default=1e-2, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    main(args)