import torch.utils.data as tud
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import datetime
import argparse
import pandas as pd
from torch.autograd import Variable
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
from Utils import *
from Model import Net
from Dataset import dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Model param
parser = argparse.ArgumentParser(description="PyTorch Spectral Compressive Imaging")
parser.add_argument('--data_path', default='./Test_data/', type=str, help='Path of data')
parser.add_argument('--mask_path', default='./mask_256_28.mat', type=str, help='Path of mask')
parser.add_argument("--size", default=256, type=int, help='The training image size')
parser.add_argument("--stage", default=3, type=str, help='Model scale')
parser.add_argument("--trainset_num", default=20000, type=int, help='The number of training samples of each epoch')
parser.add_argument("--testset_num", default=10*1, type=int, help='Total number of testset')
parser.add_argument("--seed", default=42, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=1, type=int, help='Batch_size')
parser.add_argument("--isTrain", default=False, type=bool, help='Train or test')
parser.add_argument("--bands", default=28, type=int, help='The number of channels of Datasets')
parser.add_argument("--scene_num", default=10, type=int, help='The number of scenes of Datasets')
parser.add_argument("--len_shift", default=2, type=int, help=' shift length among bands')
opt = parser.parse_args()


def prepare_testdata(file_list, file_num):
    HSI = np.zeros((((256, 256, 28, file_num))))
    for idx in range(file_num):
        path1 = file_list[idx]
        data = sio.loadmat(path1)
        HSI[:, :, :, idx] = data['img'] #/ data['img'].max()
    HSI[HSI < 0.] = 0.
    HSI[HSI > 1.] = 1.
    return HSI


def shift(f, len_shift=0):
    [bs, nC, row, col] = f.shape
    shift_f = torch.zeros(bs, nC, row, col + (nC - 1) * len_shift)
    for c in range(nC):
        shift_f[:, c, :, c * len_shift:c * len_shift + col] = f[:, c, :, :]
    return shift_f


if __name__ == "__main__":

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    print(opt)

    key = 'test_list.txt'
    file_path = opt.data_path + key
    file_list = loadpath(file_path)
    file_list.sort()
    print(file_list)
    HSI = prepare_testdata(file_list, opt.scene_num)

    Dataset = dataset(opt, HSI)
    loader_train = tud.DataLoader(Dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    # loss_func = nn.L1Loss()
    loss_func = nn.MSELoss()

    model = Net(opt).cuda()

    with torch.no_grad():
        for epoch in range(266, 300):
            checkpoint = torch.load("./Checkpoint/model_%03d.pkl" % epoch)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            epoch_loss = 0
            psnr_total = 0
            psnr_total_scene = 0
            for step, (g, label, Phi_batch, Phi_s_batch) in enumerate(loader_train):
                start = time.time()
                out = model(g, input_mask=(Phi_batch, Phi_s_batch))

                loss = loss_func(out[opt.stage-1], label)
                epoch_loss += loss.item()

                elapsed = (time.time() - start)
                out = out[opt.stage-1].clamp(min=0., max=1.)

                psnr = compare_psnr(out.detach().cpu().numpy(), label.detach().cpu().numpy(), data_range=1.0)
                psnr_total = psnr_total + psnr
                psnr_total_scene = psnr_total_scene + psnr
                if (step + 1) % (1 * 1 / opt.batch_size) == 0:
                    print(
                        '%2d psnr = %.4f' % (
                            step // (1 * 1 / opt.batch_size), psnr_total_scene / (1 * 1 / opt.batch_size)))
                    psnr_total_scene = 0

                hsi = out.detach().cpu().permute(2, 3, 1, 0).squeeze(3).numpy()
                save_path = os.path.join("./Result", '%d.mat' % (step + 1))
                sio.savemat(save_path, {'hsi': hsi})
                lbl = label.detach().cpu().permute(2, 3, 1, 0).squeeze(3).numpy()
                save_path = os.path.join("./Label", '%d.mat' % (step + 1))
                sio.savemat(save_path, {'label': lbl})

            print("model %d, test Avg PSNR = %.4f, train Avg PSNR = %.4f" % (epoch, psnr_total / (step + 1), checkpoint['psnr']))
