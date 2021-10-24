import sys
sys.path.append('../')
import os
from torch.utils.data import DataLoader, Dataset
from utils.signalprocess import wav2lps, lps2wav, wav_read
from utils.soundscape_viewer.lts_maker import lts_maker
import numpy as np
import scipy.io.wavfile as wav
from scipy.io import loadmat

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
                        LTS_run=lts_maker(sensitivity=0, channel=1, environment='wat', FFT_size=self.FFT_dict['FFTSize'], 
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
                            LTS_mean_T = LTS['mean']
                            LTS_mean_T = np.tile(LTS_mean_T[0,1:], (1, spec.shape[1]))
                            spec_T = np.reshape((spec.T), (-1,1,1,int(self.FFT_dict['FFTSize']/2+1)))[:,:,:,self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
                            LTS_mean_T = np.reshape((LTS_mean_T), (-1,1,1,int(self.FFT_dict['FFTSize']/2+1)))[:,:,:,self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
                            iter_hack = spec_T.shape[0]//4
                            for i in range(4):
                                self.samples = (spec_T[i*iter_hack:(i+1)*iter_hack,:,:,:], LTS_mean_T[i*iter_hack:(i+1)*iter_hack,:,:,:])
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
                                iter_hack = spec_T.shape[0]//4
                                for i in range(4):
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
            return (self.samples[0][index], self.samples[1][index])
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
    return lps[:, :lps.shape[1]//4], phase[:, :phase.shape[1]//4], mean, std

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

