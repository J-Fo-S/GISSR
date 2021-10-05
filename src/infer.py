# -*- coding: utf-8 -*-
# Author:Wei-Chien Wang

import sys
import torch
import numpy as np
import librosa
import scipy
import os
import scipy.io.wavfile as wav
import cv2
from utils.signalprocess import lps2wav
sys.path.append('../')

def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

def infer(net, lps, phase,  mean, std, FFT_dict, filedir=None, filename=None, args=None):
    """
        Argument:
            input: npArray. 
                log power spectrum (lps).
            phase: npArray
                to invert via griffin-lim
            mean: npArray.
                mean value of lps.
            std: npArray. 
                variance of lps.
            filedir: string. 
                directory .
            filename: string. 
                name.
    """
    #feature_dim = FFT_dict['frequency_bins'][1] - FFT_dict['frequency_bins'][0]

    #if args.model_type == "DAE_C" or "VQVAE":
    #    x = np.reshape((input.T), (-1, 1, 1, int(FFT_dict['FFTSize']/2+1)))[:, :, :,FFT_dict['frequency_bins'][0]:FFT_dict['frequency_bins'][1]]
    #else:
    #    x = input.T[:, FFT_dict['frequency_bins'][0]:FFT_dict['frequency_bins'][1]]
    #result = torch.tensor(input).float().cuda()
    
    # Inverse separated sources of log power spectrum to waveform.
    result = np.zeros(lps.shape)
    lps = np.reshape((lps.T), (-1, 1, 1, int(FFT_dict['FFTSize']/2+1)))[:, :, :,FFT_dict['frequency_bins'][0]:FFT_dict['frequency_bins'][1]]
    lps = torch.tensor(lps).cuda().float()
    if args.model_type == 'VQVAE':
        output, _ = net(lps)
    else:
        output = net(lps)
    print(output.shape)
    output = output.permute((1,2,3,0))
    output = torch.squeeze(output,0)
    output = torch.squeeze(output,0).detach().cpu()
    result[FFT_dict['frequency_bins'][0]:FFT_dict['frequency_bins'][1], :] = np.array(output[:, :])
    result = np.sqrt(10**(result*std+mean))
    result = np.multiply(result, phase)
    #Result = input
    result = librosa.istft(result, hop_length=FFT_dict['Hop_length'], win_length=FFT_dict['Win_length'], window=scipy.signal.hamming, center=False)
    result = np.int16(result*32768)
    result_path = "{0}reconstruct/".format(filedir)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    wav.write("{0}{1}.wav".format(result_path,  filename), FFT_dict['sr'], result)

# TO-DO: all below needs fix/refactor
# necessary?
class Infer(object):

    """
    Arguments:
        model: 
            deep autoencoder for source separation.
        FFT_dict: dict {'sr': int,
                        'frequency_bins': [int, int], #e.g.[0, 300]
                        'FFTSize': int,
                        'Hop_length': int,
                        'Win_length': int,
                        'normalize': bool,} 
            fouier transform parameters.
    """
    def __init__(self, model=None, FFT_dict=None, args=None):

        self.model = model
        self.args = args
    
    #TO-DO: adapt to new model/objectives
    def latent_t(self, x, mode=12):
        latent = self.model.encoder(x)
        if mode == 8:
            # prediction flow
            latent = np.uint8(latent.cpu().detach().numpy()*255)
            frame_num = latent.shape[0]
            encoding_shape = latent.shape[1]
            latent_vid = np.reshape(latent, (frame_num*8,32,32))
            latent_vid = bin_ndarray(latent_vid, (frame_num,32,32),operation='mean').astype(np.uint8)
            print(latent_vid.shape)
            img = np.array(latent_vid)
            frameSize = (32, 32)
            out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize, 0)
            for i in range(0,frame_num):
                #print(i)
                #img = np.ones((500, 500, 3), dtype=np.uint8)*i
                out.write(img[i,:,:])
            out.release()
            exit()
        if mode == 9:
            # b-frame flow
            latent = latent.cpu().detach().numpy()*255
            frame_num = latent.shape[0]
            encoding_shape = latent.shape[1]
            latent_vid = np.reshape(latent, (frame_num*8,32,32))
            latent_vid = bin_ndarray(latent_vid, (frame_num,32,32),operation='mean').astype(np.uint8)
            print(latent_vid.shape)
            img = np.array(latent_vid)
            frameSize = (32, 32)
            out = cv2.VideoWriter('output_video_time_prev.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize, 0)
            for i in range(1,frame_num):
                #print(i)
                #img = np.ones((500, 500, 3), dtype=np.uint8)*i
                out.write(np.uint8(np.log10((img[i,:,:]**2)/(1e-8+img[i-1,:,:]**2))))
            out.release()
            exit()
        if mode == 10:
            #b-frame flow smoother representation if *255 placed after log division
            latent = latent.cpu().detach().numpy()
            frame_num = latent.shape[0]
            encoding_shape = latent.shape[1]
            latent_vid = np.reshape(latent, (frame_num*8,32,32))
            latent_vid = bin_ndarray(latent_vid, (frame_num,32,32),operation='mean').astype(np.uint8)
            print(latent_vid.shape)
            img = np.array(latent_vid)
            frameSize = (32, 32)
            out = cv2.VideoWriter('output_video_time_prev_decim.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize, 0)
            for i in range(1,frame_num):
                #print(i)
                #img = np.ones((500, 500, 3), dtype=np.uint8)*i
                out.write(np.uint8(np.log10((img[i,:,:]**2)/(1e-8+img[i-1,:,:]**2))*255))
            out.release()
            exit()
        if mode == 11:
            # lts flow
            latent = latent.cpu().detach().numpy()
            frame_num = latent.shape[0]
            encoding_shape = latent.shape[1]
            latent_vid = np.reshape(latent, (frame_num*8,32,32))
            latent_vid = bin_ndarray(latent_vid, (frame_num,32,32),operation='mean').astype(np.uint8)
            print(latent_vid.shape)
            img = np.array(latent_vid)
            frameSize = (32, 32)
            out = cv2.VideoWriter('output_video_cumave_lts.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize, 0)
            # imagine this is the initial LTS
            lts = np.mean(img, axis=0)
            # and these are the LTS flow of generated specs
            # this adjusts the influence of global structure or local flow
            global_local = 0.01
            for i in range(0,frame_num):
                #print(i)
                #img = np.ones((500, 500, 3), dtype=np.uint8)*i
                flow_control = frame_num * global_local
                lts = lts*((flow_control-1)/flow_control) + img[i]*(1/flow_control)
                out.write(np.uint8((lts/np.max(lts))*255))
            out.release()
            exit()
        if mode == 12:
            # lts difference flow
            latent = latent.cpu().detach().numpy()*255
            frame_num = latent.shape[0]
            encoding_shape = latent.shape[1]
            latent_vid = np.reshape(latent, (frame_num*8,32,32))
            latent_vid = bin_ndarray(latent_vid, (frame_num,32,32),operation='mean').astype(np.uint8)
            print(latent_vid.shape)
            img = np.array(latent_vid)
            frameSize = (32, 32)
            out = cv2.VideoWriter('output_video_cumave_lts_diff.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize, 0)
            # imagine this is the initial LTS
            lts = np.mean(img, axis=0)
            # and these are the LTS flow of generated specs
            # this adjusts the influence of global structure or local flow
            global_local = 0.01
            for i in range(1,frame_num):
                #print(i)
                #img = np.ones((500, 500, 3), dtype=np.uint8)*i
                flow_control = frame_num * global_local
                lts = np.log10((lts*((flow_control-1)/flow_control) + img[i]*(1/flow_control))**2/(1e-8+lts*((flow_control-1)/flow_control) + img[i-1]*(1/flow_control))**2)
                out.write(np.uint8((lts/np.max(lts))))
            out.release()
            exit()
        # latent target, top feature, bottom feature, decoded output
        return latent, lts_g, lts_l, spec_out

    def infer(self, input, mean, std, filedir, filename):
        """
            Argument:
                input: npArray. 
                    log power spectrum (lps).
                mean: npArray.
                    mean value of lps.
                std: npArray. 
                    variance of lps.
                filedir: string. 
                    directory .
                filename: string. 
                    name.
        """
        feature_dim = self.FFT_dict['frequency_bins'][1] - self.FFT_dict['frequency_bins'][0]

        if self.args.model_type == "DAE_C":
            x = np.reshape((input.T), (-1, 1, 1, int(self.FFT_dict['FFTSize']/2+1)))[:, :, :,self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
        else:
            x = input.T[:, self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
        x = torch.tensor(x).float().cuda()
        
        # Inverse separated sources of log power spectrum to waveform.
        input = np.sqrt(10**(input*std+mean))

        Result = np.array(input)
        print(Result.shape)
        result = librosa.griffinlim(Result[self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1], :], hop_length=self.FFT_dict['Hop_length'], win_length=self.FFT_dict['Win_length'], window=scipy.signal.hamming, center=True)
        result = np.int16(result*32768)
        result_path = "{0}reconstruct/".format(filedir)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        wav.write("{0}{1}.wav".format(result_path,  filename), self.FFT_dict['sr'], result)