# -*- coding: utf-8 -*-
#Author: Wei-Chien Wang
"""
Created on Fri Oct 16 10:24:51 2020
This code is used for unsupervised blind monoural periodic source separation based on perioidic-coded deep autoencoder (, with MSE loss)
If you find this code useful in your research, please cite:
Citation: 
       [1] K.-H. Tsai, W.-C. Wang, C.-H. Cheng, C.-Y. Tsai, J.-K. Wang, T.-H. Lin, S.-H. Fang, L.-C. Chen, and Y. Tsao, "Blind Monaural Source Separation on Heart and Lung Sounds Based on Periodic-Coded Deep Autoencoder," to appear in IEEE Journal of Biomedical and Health Informatics.

Contact:
       Wei-Chien Wang
       Weichian0920@gmail.com
       Academia Sinica, Taipei, Taiwan
       
"""
import argparse
import time
from utils import misc
import torch
from torch.autograd import Variable
from torchinfo import summary
from datetime import datetime
import numpy as np
from model import DAE_C, DAE_F, VQVAE, VQVAE_C
import train_ae, train_vae, infer
from source_separation import MFA
from dataset.HLsep_dataloader import hl_dataloader, val_dataloader, prewhiten
from utils.soundscape_viewer.lts_maker import lts_maker
import scipy.io.wavfile as wav
from scipy.io import loadmat
import os
import sys

# parser#

parser = argparse.ArgumentParser(description='PyTorch Source Separation')
parser.add_argument('--model_type', type=str, default='DAE_C', help='model type', choices=['DAE_C', 'DAE_F', 'VQVAE', 'VQVAE_C'])
parser.add_argument('--data_feature', type=str, default='lps', help='lps or wavform')
parser.add_argument('--pretrained', dest='pretrained', default=False, action='store_true', help='load pretrained model or not')
parser.add_argument('--preproc_lts', dest='preproc_lts', default=False, action='store_true', help='use if need to preprocess lts files')
parser.add_argument('--pretrained_path', type=str, default="log/default/VQVAE_20211001_1627.pth", help='pretrained_model path')
# training hyperparameters
parser.add_argument('--optim', type=str, default="Adam", help='optimizer for training', choices=['RMSprop', 'SGD', 'Adam'])
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for training (default: 1e-3)')
parser.add_argument('--CosineAnnealingWarmRestarts', type=bool, default=False, help='optimizer scheduler for training')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default/', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='200,250', help='decreasing strategy')
parser.add_argument('--prewhiten', type=int, default=10, help='backgroud noise removal')
# MFA hyperparameters
parser.add_argument('--source_num', type=int, default=3, help='number of separated sources')
parser.add_argument('--clustering_alg', type=str, default='NMF', choices=['NMF', 'K_MEANS', 'seqNMF', 'seqNMF_WEIGHTS'], help='clustering algorithm for embedding space')
parser.add_argument('--wienner_mask', type=bool, default='store_true', help='wienner time-frequency mask for output')
#VQ-VAE training parameters
parser.add_argument("--n_gpu", type=int, default=1)
port = (
    2 ** 15
    + 2 ** 14
    + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
)
parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
#TO-DO: fix size - add own batch size/lr (using above now)
parser.add_argument('--size', type=int, default=1024)
# REDUNDANT
#parser.add_argument('--epoch', type=int, default=50)
#parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--sched', type=str, help='cycle')
parser.add_argument('--path', type=str, default='log/default')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
misc.logger.init(args.logdir, 'train_log_')
logger = misc.logger.info

starttime = time.time()
current_time = datetime.now().strftime('%Y%m%d_%H%M')
args.logdir = args.logdir + str(args.model_type) + "_" + str(current_time)

misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
    logger('{}: {}'.format(k, v))
logger("========================================")

"""
if args.seed is not None:
    random.seed(args.seed)
    cudnn.deterministic=None
    ngpus_per_node = torch.cuda.device_count()
"""
# build model
#decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
#logger('decreasing_lr: ' + str(decreasing_lr))
best_acc, old_file = 0, None
per_save_epoch = 30
t_begin = time.time()
grad_scale = args.grad_scale

# Default model dictionary
DAE_C_dict = {
        "frequency_bins": [0, 1024],
        "encoder": [32, 16, 8],
        "decoder": [8, 16, 32, 1],
        "encoder_filter": [[1, 3], [1, 3], [1, 3]],
        "decoder_filter": [[1, 3], [1, 3], [1, 3], [1, 1]],
        "encoder_act": "relu",
        "decoder_act": "relu",
        "dense": [],
        }

DAE_F_dict = {
        "frequency_bins": [0, 1024],
        "encoder": [1024, 512, 256, 128],
        "decoder": [256, 512, 1024, 1025],
        "encoder_act": "relu",
        "decoder_act": "relu",
        }

#TO-DO: make model dict for VQ-VAE
VQVAE_dict = {}

VQVAE_C_dict = {}

Model = {
    'DAE_C': DAE_C.autoencoder,
    'DAE_F': DAE_F.autoencoder,
    'VQVAE': VQVAE.VQVAE,
    'VQVAE_C': VQVAE_C.VQVAE_C
}

model_dict = {
    'DAE_C': DAE_C_dict,
    'DAE_F': DAE_F_dict,
    'VQVAE': VQVAE_dict,
    'VQVAE_C': VQVAE_C_dict
}


# Default fourier transform parameters
FFT_dict = {
    'sr': 44100,
    'frequency_bins': [0, 1024],
    'FFTSize': 2048,
    'Hop_length': 128,
    'Win_length': 2048,
    'normalize': True,
}
# declare model object
net = Model[args.model_type](model_dict=model_dict[args.model_type], args=args, logger=logger).cuda()

torch.manual_seed(args.seed)

# remove for vae?
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    net.cuda()

#summary(net, input_size=(args.batch_size, 1, 1, args.size))
#print(torch.cuda.memory_summary(device=None, abbreviated=False))

if __name__ == "__main__":
    if args.pretrained == False:
        # data loader
        print(os.getcwd())
        #if args.model_type == 'VQVAE':
        data_path_list = 'src/data/sesoko/Audio'
        #else:
        #    test_filelist = ["src/data/sesoko/Audio/B20/SSK_Site_B_20170601_174000.wav"]
        test_filename = "0_0"
        outdir = "{0}/test_".format(args.logdir)
        train_loader = hl_dataloader(data_path_list, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, FFT_dict=FFT_dict, args=args)
        # train
        if args.model_type == 'VQVAE' or 'VQVAE_C':
            net = train_vae.train_vae(train_loader, net, args, logger)
        else:
            net = train_ae.train_ae(train_loader, net, args, logger)
        # NOTE: just a 'test' to output very same data as result?
        #for batch_idx, data in enumerate(train_loader):
        #    if args.cuda:
        #        data = data.cuda().float()
        #    data = Variable(data)
        #    output = net(data)
        #    output_ = torch.reshape(output[0], (-1,)).detach().cpu()
        #output_ = output_.numpy()
        #wav.write("reconstruct.wav", 8000, np.int16(output_*32768.))

        # reconstruction test
        net.eval()
        with torch.no_grad():
            for subdir, _, files in os.walk(data_path_list):
                if files:
                    if args.model_type == "VAEVQ_C" and not args.preproc_lts:
                        LTS_run=lts_maker(sensitivity=0, channel=1, environment='wat', FFT_size=FFT_dict['FFTSize'], 
                            window_overlap=FFT_dict['Hop_length']/FFT_dict['FFTSize'], initial_skip=0)
                        LTS_run.collect_folder(path=subdir)
                        LTS_run.filename_check(dateformat='yyyymmdd_HHMMSS',initial='1207984160.',year_initial=2000)
                        LTS_run.run(save_filename='separate', folder_id=subdir)
                    for filename in files:
                        if filename.endswith('.wav'):
                            filepath = os.path.join(subdir, filename)  
                            # load test data
                            spec, phase, mean, std = val_dataloader(filepath, FFT_dict, args=args)
                            print(spec.shape)
                            if args.prewhiten > 0: 
                                spec, _ = prewhiten(spec, args.prewhiten, 0)
                                spec[spec<0] = 0
                            if not os.path.exists(filepath[:-4]+'.mat'):
                                print("preprocess LTS first!")
                                exit()
                            LTS = loadmat(filepath[:-4]+'.mat')
                            LTS_mean = LTS['mean']
                            LTS_mean = np.tile(LTS_mean[0,1:], (1, spec.shape[1]))
                            #bc griffin-lim inversion needs fft/2+1
                            if args.model_type == "VQVAE_C":
                                spec = (spec, LTS_mean)
                            infer.infer(net, spec, phase, np.array(mean), np.array(std), FFT_dict, filedir=outdir, filename=test_filename, args=args)
                            exit()
    # TO-DO: fix val_dataloader for proper data iteration
    else:
        net.load_state_dict(torch.load(args.pretrained_path))
        # data loader
        print(os.getcwd())
        test_filelist = ["src/data/sesoko/Audio/B20/SSK_Site_B_20170601_174000.wav"]
        test_filename = "0_0"
        outdir = "{0}/test_".format(args.logdir)
        #with torch.no_grad():
        #    for test_file in test_filelist:
        #        # load test data
        #        lps, _, mean, std = val_dataloader(test_file, FFT_dict, args=args)
        #        infer.infer(np.array(lps), np.array(mean), np.array(std), FFT_dict, filedir=outdir, filename=test_filename, args=args)
        net.eval()
        with torch.no_grad():
            for test_file in test_filelist:
                # load test data
                lps, phase, mean, std = val_dataloader(test_file, FFT_dict, args=args)
                print(lps.shape)
                #bc griffin-lim inversion needs fft/2+1
                infer.infer(net, lps, phase, np.array(mean), np.array(std), FFT_dict, filedir=outdir, filename=test_filename, args=args)
