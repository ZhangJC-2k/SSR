import torch.utils.data as tud
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import datetime
import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn
from Utils import *
from Model import Net
from Dataset import dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Model Config
parser = argparse.ArgumentParser(description="PyTorch Spectral Compressive Imaging")
parser.add_argument('--data_path', default='./CAVE_1024_28/', type=str, help='Path of data')
parser.add_argument('--mask_path', default='./mask_256_28.mat', type=str, help='Path of mask')
parser.add_argument("--size", default=256, type=int, help='The training image size')
parser.add_argument("--stage", default=9, type=str, help='Model scale')
parser.add_argument("--trainset_num", default=5000, type=int, help='The number of training samples of each epoch')
parser.add_argument("--testset_num", default=5, type=int, help='Total number of testset')
parser.add_argument("--seed", default=42, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=2, type=int, help='Batch_size')
parser.add_argument("--isTrain", default=True, type=bool, help='Train or test')
parser.add_argument("--bands", default=28, type=int, help='The number of channels of Datasets')
parser.add_argument("--scene_num", default=205, type=int, help='The number of scenes of Datasets') ## 205 
parser.add_argument("--lr", default=0.0004, type=float, help='learning rate')
parser.add_argument("--len_shift", default=2, type=int, help=' shift length among bands')
opt = parser.parse_args()


def loss_f(loss_func, pred, lbl):
    return torch.sqrt(loss_func(pred, lbl))


if __name__ == "__main__":

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    print(opt)

    model = Net(opt)
    model.cuda()

    print('time = %s' % (datetime.datetime.now()))
    ## Load training data
    key = 'train_list.txt'
    file_path = opt.data_path + key
    file_list = loadpath(file_path)
    file_list.sort()
    HSI = prepare_data(file_list, opt.scene_num)

    Dataset = dataset(opt, HSI)
    loader_train = tud.DataLoader(Dataset, batch_size=opt.batch_size, shuffle=True)
    print('time = %s' % (datetime.datetime.now()))

    # loss_func = nn.L1Loss()
    mse = torch.nn.MSELoss().cuda()
    ## Load trained model
    start_epoch = findLastCheckpoint(save_dir="./Checkpoint")
    if start_epoch > 0:
        print('Load model: resuming by loading epoch %03d' % start_epoch)
        checkpoint = torch.load(os.path.join("./Checkpoint", 'model_%03d.pkl' % start_epoch))
        model.load_state_dict(checkpoint['model'])
        start_epoch = 1 + checkpoint['epoch']
    else:
        start_epoch = 1
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': opt.lr}], lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=1e-6, last_epoch=start_epoch - 2)
    model.train()
    print('time = %s' % (datetime.datetime.now()))
    ## pipline of training
    for epoch in range(300):
        epoch_loss = 0
        psnr_total = 0
        start_time = time.time()
        for step, (g, label, Phi_batch, Phi_s_batch) in enumerate(loader_train):

            out = model(g=g, input_mask=(Phi_batch, Phi_s_batch))

            psnr = compare_psnr(label.detach().cpu().numpy(), out[opt.stage-1].detach().cpu().numpy(), data_range=1.0)
            psnr_total = psnr_total + psnr

            loss = loss_f(mse, out[opt.stage-1], label) + 0.7 * loss_f(mse, out[opt.stage-2], label) + \
                   0.5 * loss_f(mse, out[opt.stage-3], label) + 0.3 * loss_f(mse, out[opt.stage-4], label)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print('%4d %4d / %4d loss = %.16f time = %s' % (
                    start_epoch + epoch, step, len(Dataset) // opt.batch_size,
                    epoch_loss / ((step + 1) * opt.batch_size),
                    datetime.datetime.now()))

        elapsed_time = time.time() - start_time
        scheduler.step()
        print('epoch = %4d , loss = %.16f , Avg PSNR = %.4f ,time = %4.2f s' % (
            start_epoch + epoch, epoch_loss / len(Dataset), psnr_total / (step + 1), elapsed_time))
        state = {'model': model.state_dict(), 'epoch': start_epoch + epoch,
                 'loss': epoch_loss / len(Dataset), 'psnr': psnr_total / (step + 1)}
        torch.save(state, os.path.join("./Checkpoint", 'model_%03d.pkl' % (start_epoch + epoch)))
