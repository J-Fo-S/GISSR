from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import librosa.display
import sys
sys.path.append('../')
import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T
from utils.signalprocess import wav2lps, wav_read
from utils.soundscape_viewer.lts_maker import lts_maker
import numpy as np
import scipy.io.wavfile as wav
from scipy.io import loadmat


def save_test(out, filename='test.png'):
    # For plotting headlessly
    #fig = plt.Figure()
    #canvas = FigureCanvas(fig)
    #ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time', sr=44100, hop_length=128)
    fig.colorbar(p, ax=ax, format="%+2.f dB")
    fig.savefig(filename)
    plt.close(fig)

class HL_dataset(Dataset):

    def __init__(self, data_path_list, FFT_dict, args):

        self.FFT_dict = FFT_dict
        self.args  = args
        # see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        # and here for map-style data loading https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
        self.data_path_list =  data_path_list
        if args.data_feature=="lps_lts":
            for subdir, _, files in os.walk(self.data_path_list):
                print(f'{subdir} {files}')
                if files: 
                    if not args.preproc_lts:
                        LTS_run=lts_maker(sensitivity=-60, channel=1, environment='wat', FFT_size=self.FFT_dict['FFTSize'], 
                            window_overlap=self.FFT_dict['Hop_length']/self.FFT_dict['FFTSize'], initial_skip=0)
                        LTS_run.collect_folder(path=subdir)
                        LTS_run.filename_check(dateformat='yyyymmdd_HHMMSS',initial='1207984160.',year_initial=2000)
                        LTS_run.run(save_filename='separate', folder_id=subdir)
                    for filename in files:
                        if filename.endswith('.wav'):
                            filepath = os.path.join(subdir, filename)  
                            spec, _, _, _ = wav2lps(filepath, self.FFT_dict['FFTSize'],  self.FFT_dict['Hop_length'],  self.FFT_dict['Win_length'],  self.FFT_dict['normalize'])
                            if args.prewhiten > 0: 
                                spec, _ = prewhiten(spec, args.prewhiten, 0)
                                spec[spec<0] = 0
                            if not os.path.exists(filepath[:-4]+'.mat'):
                                print("preprocess LTS first!")
                                exit()
                            LTS = loadmat(filepath[:-4]+'.mat')
                            LTS_mean = LTS['mean']
                            LTS_mean = np.tile(LTS_mean[0,1:], (spec.shape[1], 1)).T
                            # ugly tensor/np conversions to avoid tensor enumerate problem
                            spec_targ = spec.copy()
                            spec = torch.cuda.FloatTensor(spec)
                            spec_targ = torch.cuda.FloatTensor(spec_targ)
                            LTS_mean = torch.cuda.FloatTensor(LTS_mean)
                            iter_hack = spec.shape[1]//8
                            for i in range(8):
                                masking = T.FrequencyMasking(freq_mask_param=80, iid_masks=False)
                                #masking_match = T.FrequencyMasking(freq_mask_param=80, iid_masks=False)
                                randint = np.random.randint(0, high=1000)
                                randintb = np.random.randint(0, high=1000)
                                torch.random.manual_seed(randint)
                                for _ in range(4):
                                    LTS_mean[:,i*iter_hack:(i+1)*iter_hack] = masking(LTS_mean[:,i*iter_hack:(i+1)*iter_hack])
                                masking = T.FrequencyMasking(freq_mask_param=80, iid_masks=False)
                                torch.random.manual_seed(randintb)
                                for _ in range(4):
                                    spec[:,i*iter_hack:(i+1)*iter_hack] = masking(spec[:,i*iter_hack:(i+1)*iter_hack])
                                masking = T.FrequencyMasking(freq_mask_param=80, iid_masks=False)
                                torch.random.manual_seed(randint)
                                for _ in range(4):
                                    spec_targ[:,i*iter_hack:(i+1)*iter_hack] = masking(spec_targ[:,i*iter_hack:(i+1)*iter_hack])
                            save_test(LTS_mean.cpu().numpy(), filename=self.args.logdir+'/lts_train.png')
                            save_test(spec.cpu().numpy(), filename=self.args.logdir+'/lps_train.png')
                            save_test(spec_targ.cpu().numpy(), filename=self.args.logdir+'/lps_targ.png')
                            spec_T = torch.reshape((spec.T), (-1,1,1,int(self.FFT_dict['FFTSize']/2+1)))[:,:,:,self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
                            spec_targ_T = torch.reshape((spec_targ.T), (-1,1,1,int(self.FFT_dict['FFTSize']/2+1)))[:,:,:,self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
                            LTS_mean_T = torch.reshape((LTS_mean.T), (-1,1,1,int(self.FFT_dict['FFTSize']/2+1)))[:,:,:,self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
                            #exit()
                            # NOTE: use for OOM errors so don't have to rewrite dataloader
                            spec_T = spec_T.detach().cpu().numpy()
                            spec_targ_T = spec_targ_T.detach().cpu().numpy()
                            LTS_mean_T = LTS_mean_T.detach().cpu().numpy()
                            for i in range(8):
                                self.samples = (spec_T[i*iter_hack:(i+1)*iter_hack,:,:,:], LTS_mean_T[i*iter_hack:(i+1)*iter_hack,:,:,:], spec_targ_T[i*iter_hack:(i+1)*iter_hack,:,:,:])
        elif args.data_feature=="lps":
            for subdir, _, files in os.walk(self.data_path_list):
                print(f'{subdir} {files}')
                if files:
                    for filename in files:
                        if filename.endswith('.wav'):
                            filepath = os.path.join(subdir, filename)  
                            spec, _, _, _ = wav2lps(filepath, self.FFT_dict['FFTSize'],  self.FFT_dict['Hop_length'],  self.FFT_dict['Win_length'],  self.FFT_dict['normalize'])
                            if args.prewhiten > 0: 
                                spec, _ = prewhiten(spec, args.prewhiten, 0)
                                spec[spec<0] = 0
                            if args.model_type=="DAE_C" or "VQVAE":
                                spec_T = np.reshape((spec.T), (-1,1,1,int(self.FFT_dict['FFTSize']/2+1)))[:,:,:,self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
                                iter_hack = spec_T.shape[0]//8
                                for i in range(8):
                                    self.samples = spec_T[i*iter_hack:(i+1)*iter_hack,:,:,:]
                            else:
                                self.samples = spec.T[:,self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
        else:
            for subdir, _, files in os.walk(self.data_path_list):
                for filename in files:
                    filepath = os.path.join(subdir, filename)  
                    spec, _, _, _ = wav2lps(filepath, self.FFT_dict['FFTSize'],  self.FFT_dict['Hop_length'],  self.FFT_dict['Win_length'],  self.FFT_dict['normalize'])
                    if args.prewhiten > 0: 
                        spec, _ = prewhiten(spec, args.prewhiten, 0)
                        spec[spec<0] = 0
                    y = wav_read(filepath)
                    self.samples = np.reshape(y, (-1,1,1,y.shape[0]))


    def __getitem__(self, index):
        if self.args.data_feature=="lps_lts":
            return (self.samples[0][index], self.samples[1][index], self.samples[2][index])
        else:
            return self.samples[index]


    def __len__(self):
        if self.args.data_feature=="lps_lts":
            return len(self.samples[0])
        else:
            return len(self.samples)

def hl_dataloader(data_path_list, batch_size=311, shuffle=False, num_workers=1, pin_memory=True, FFT_dict=None, args=None):

    hl_dataset = HL_dataset(data_path_list, FFT_dict, args)
    # TO-DO: shuffle true here and False in dataset? Leave for now.. 
    # see: https://discuss.pytorch.org/t/dataloader-just-shuffles-the-order-of-batches-or-does-it-also-shuffle-the-images-in-each-batch/60900/11
    hl_dataloader = DataLoader(hl_dataset, batch_size = batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return hl_dataloader

def val_dataloader(filepath, FFT_dict, args=None):
    # TO-DO: add dataloader for inference memory management
    lps, phase, mean, std = wav2lps(filepath, FFT_dict['FFTSize'], FFT_dict['Hop_length'], FFT_dict['Win_length'], FFT_dict['normalize'])
    if args.prewhiten > 0: 
        lps, _ = prewhiten(lps, args.prewhiten, 0)
        lps[lps<0] = 0
        lps = np.array(lps)
        phase = np.array(phase)
    return lps[:, :lps.shape[1]//8], phase[:, :phase.shape[1]//8], mean, std

def prewhiten(input_data, prewhiten_percent, axis):
        import numpy.matlib
        list=np.where(np.abs(input_data)==float("inf"))[0]
        input_data[list]=float("nan")
        input_data[list]=np.nanmin(input_data)
        
        ambient = np.percentile(input_data, prewhiten_percent, axis=axis)
        if axis==0:
            input_data = np.subtract(input_data, np.matlib.repmat(ambient, input_data.shape[axis], 1))
        elif axis==1:
            input_data = np.subtract(input_data, np.matlib.repmat(ambient, input_data.shape[axis], 1).T)
        return input_data, ambient

