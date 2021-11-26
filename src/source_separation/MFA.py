# -*- coding: utf-8 -*-
# Author:Wei-Chien Wang

import sys
import torch
import numpy as np
import librosa
import scipy
from utils.clustering_alg import K_MEANS,  NMF_clustering
import os
import scipy.io.wavfile as wav
import cv2
from utils.signalprocess import lps2wav
from seqnmf import seqnmf
from seqnmf import helpers
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

class MFA_source_separation(object):

    """
    MFA analysis for unsupervised monoaural blind source separation.
        This function separate different sources by unsupervised manner depend on different source's periodicity properties.
    Arguments:
        model: 
            deep autoencoder for source separation.
        source number: int.
            separated source quantity. type: int.
        clustering_alg: string. 
            "NMF" or "K_MEANS". clustering algorithm for MFA analysis. . type: str 
        wienner_mask: bool. 
            if True the output is mask by constructed ratio mask.
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
        self.source_num = args.source_num
        self.clustering_alg = args.clustering_alg
        self.wienner_mask = args.wienner_mask
        self.FFT_dict = FFT_dict
        self.args = args
    
    def FFT_(self, input):
        epsilon = np.finfo(float).eps
        frame_num = input.shape[1]
        encoding_shape = input.shape[0]
        print(frame_num)
        print(encoding_shape)
        FFT_result = np.zeros((encoding_shape, int(frame_num/2+1)))

        for i in range(0, encoding_shape):
            fft_r = librosa.stft(input[i, :], n_fft=frame_num, hop_length=frame_num+1, window=scipy.signal.hamming)
            fft_r = fft_r+ epsilon
            FFT_r = abs(fft_r)**2
            FFT_result[i] = np.reshape(FFT_r, (-1,))

        return FFT_result
    
    def weight_nmf(self, x, w_s, sources, mode=12):
        #w_s = weight_layer
        #print(w_s[:2,:2,:,:])
        w_s = np.squeeze(w_s, axis=2)
        shape0 = w_s.shape[0]
        shape1 = w_s.shape[1]
        shape3 = w_s.shape[2]
        # transpose colums to rows - zero-out negatives
        if mode == 1:
            w_s = w_s.transpose(0,2,1).reshape(w_s.shape[0],-1)
            w_s[w_s < 0] = 0
            print(w_s.shape)
        # arrange kernels side-by-side column-wise - zero-out negatives
        if mode == 2:
            w_s = np.hstack((w_s[:,:,0], w_s[:,:,1], w_s[:,:,2]))
            w_s[w_s < 0] = 0
            print(w_s.shape)
        # arrange kernels vertically row-wise - zero-out negatives
        if mode == 3:
            w_s = np.vstack((w_s[:,:,0], w_s[:,:,1], w_s[:,:,2]))
            w_s[w_s < 0] = 0
            print(w_s.shape)
        # arrange kernels side-by-side column-wise - flip negatives, concat and re-merge
        if mode == 4:
            w_s_holder = np.hstack((w_s[:,:,0], w_s[:,:,1], w_s[:,:,2]))
            w_s = np.hstack((np.maximum(0, w_s), np.maximum(0, -w_s)))
            print(w_s.shape)
            w_s = np.hstack((w_s[:,:,0], w_s[:,:,1], w_s[:,:,2]))
            print(w_s.shape)
        # NOT WORKING like 4 but row-wise
        if mode == 5:
            w_s_holder = np.vstack((w_s[:,:,0], w_s[:,:,1], w_s[:,:,2]))
            w_s = np.vstack((np.maximum(0, w_s), np.maximum(0, -w_s)))
            print(w_s.shape)
            w_s = np.vstack((w_s[:,:,0], w_s[:,:,1], w_s[:,:,2]))
            print(w_s.shape)
        # use absolute weights
        #if mode == 5:
        #    w_s_holder = np.ones((shape0, shape1, shape3))
        #    np.where(w_s < 0, w_s_holder, w_s_holder*-1)
        #    w_s = np.hstack((w_s[:,:,0], w_s[:,:,1], w_s[:,:,2]))
        #    w_s = np.abs(w_s)
        #    print(w_s.shape)
        #    print(w_s_holder[:2,:2])
        # like mode 4, but pos on one side negs on the other (e.g. 48 pos then 48 neg)
        if mode == 6:
            w_s_holder = np.hstack((w_s[:,:,0], w_s[:,:,1], w_s[:,:,2]))
            w_s = np.hstack((w_s[:,:,0], w_s[:,:,1], w_s[:,:,2]))
            print(w_s.shape)
            w_s = np.hstack((np.maximum(0, w_s), np.maximum(0, -w_s)))
            print(w_s.shape)
        # like mode 6 but vertically arranged
        if mode == 7:
            w_s_holder = np.vstack((w_s[:,:,0], w_s[:,:,1], w_s[:,:,2]))
            w_s = np.vstack((w_s[:,:,0], w_s[:,:,1], w_s[:,:,2]))
            print(w_s.shape)
            w_s = np.vstack((np.maximum(0, w_s), np.maximum(0, -w_s)))
            print(w_s.shape)
        print(w_s[:2,:2])
        #latents = torch.unsqueeze(torch.zeros_like(x), 0)
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
        sources[0] = self.model.decoder(latent)
        sources = torch.cat((sources, torch.unsqueeze(self.model.decoder(latent), 0)), 0)
        W, H, cost, loading, power = seqnmf(w_s, K=self.source_num, L=3, Lambda=0.001, lambda_L1W=0, lambda_OrthW=0.1,\
            max_iter=1000)
        #BUG?
        #for i in range(0, 1000):
        #    W, H, cost, loading, power = seqnmf(w_s, K=self.source_num, L=3, Lambda=0.001, lambda_L1W=0, lambda_OrthH=0,\
        #        W_init=W, H_init=H, max_iter=1, sort_factors=0)
        
        for source_idx in range(0, self.source_num):
            H_k = np.array(H)
            #H_k[:,:] = np.min(H)
            H_k[:,:] = 0
            for i in range (0, H.shape[1]):
                if (H_k[source_idx,i] == np.argmax(H[:,i])):
                    H_k[source_idx,i] = H[source_idx,i]

            W_k = np.array(W)
            #W_k[:,:,:] = np.min(W)
            W_k[:,:,:] = 0
            for i in range (0, W.shape[0]):
                for j in range (0, W.shape[2]):
                    if (W_k[i,source_idx,j] == np.argmax(W[i,:,j])):
                        W_k[i,source_idx,j] = W[i,source_idx,j]
            #W_k = W
            #W_k[:,:,:] = 0
            #W_k[:,source_idx,:] = W[:,source_idx,:]
            w_s = helpers.reconstruct(W_k,H_k)
            print(w_s[:2,:2])
            if mode == 4:
                w_s_copy = w_s
                w_s = np.zeros((shape0, shape1*3))
                for i in range(0, shape1*3, shape1):
                    print(i)
                    w_s[:, i:i+shape1] = w_s_copy[:, 2*i:2*i+shape1]
                    np.where(w_s_holder[:, i:i+shape1] < 0, w_s[:, i:i+shape1], -1*w_s_copy[:, 3*shape1+i:3*shape1+i+shape1])
                print(w_s.shape)
            if mode == 5:
                w_s_copy = w_s
                w_s = np.zeros((shape0*3, shape1))
                for i in range(0, shape0*3, shape0):
                    print(i)
                    w_s[i:i+shape0, :] = w_s_copy[2*i:2*i+shape0, :]
                    np.where(w_s_holder[i:i+shape0, :] < 0, w_s[i:i+shape0, :], -1*w_s_copy[3*shape0+i:3*shape0+i+shape0, :])
                print(w_s.shape)
            if mode == 6:
                w_s_copy = w_s
                w_s = np.zeros((shape0, shape1*3))
                for i in range(0, shape1*3, shape1):
                    print(i)
                    w_s[:, i:i+shape1] = w_s_copy[:, i:i+shape1]
                    np.where(w_s_holder[:, i:i+shape1] < 0, w_s[:, i:i+shape1], -1*w_s_copy[:, 2*i+shape1:2*i+2*shape1])
                print(w_s.shape)
            if mode == 7:
                w_s_copy = w_s
                w_s = np.zeros((shape0*3, shape1))
                for i in range(0, shape0*3, shape0):
                    print(i)
                    w_s[i:i+shape0, :] = w_s_copy[i:i+shape0, :]
                    np.where(w_s_holder[i:i+shape0, :] < 0, w_s[i:i+shape0, :], -1*w_s_copy[2*i+shape0:2*i+2*shape0, :])
                print(w_s.shape)
            w_s = np.reshape(w_s, (shape0, shape1, shape3))
            if mode == 5:
                w_s = w_s * w_s_holder
            w_s = w_s[:,:,np.newaxis,:]
            print(w_s[:2,:2,:,:])
            self.model.encoder.conv_layers[-2].weight = torch.nn.parameter.Parameter(torch.from_numpy(w_s).float().cuda())
            latent_code_nmf = self.model.encoder(x)
            sources = torch.cat((sources, torch.unsqueeze(self.model.decoder(latent_code_nmf), 0)), 0)
        return sources

    def freq_modulation(self, source_idx, label, encoded_img):
        """
          This function use to modulate latent code.
          Arguments:
            source_idx: int.
                source quantity. 
            Encoded_img: Tensor. 
                latent coded matrix. Each dimension represnet as (encoding_shape, frame_num)
            label: int 
                latent neuron label.
        """
        frame_num = encoded_img.shape[0]
        encoding_shape = encoded_img.shape[1]
        # minimun value of latent unit.
        min_value = torch.min(encoded_img)
        #max_inv = torch.max(encoded_img)

        for k in range(0, encoding_shape):
            if(label[k] != source_idx):
                # deactivate neurons 
                encoded_img[:,k] = min_value  # (encoding_shape,frame_num)
                #encoded_img[:,k] = -max_inv  # (encoding_shape,frame_num)
                #encoded_img[:,k] = 0  # (encoding_shape,frame_num)
        return encoded_img


    def MFA(self, input, source_num=3):
        """
          Modulation Frequency Analysis of latent space.
          Note: Each dimension of input is (frame number, encoded neuron's number).
          Arguments:
              input: 2D Tensor.
              source_num: int.
                  source quantity.
              
        """
        encoded_dim  = input.shape[1]
        # Period clustering
        #fft_bottleneck = self.FFT_(input.T)#(fft(encoded, frame_num))
        #W = np.array(fft_bottleneck[:, 3:]).T
        #H = np.array(fft_bottleneck[:, 3:])
        #print(W.shape)
        #print(H.shape)
        if self.clustering_alg == "K_MEANS":
            k_labels, k_centers = K_MEANS.create_cluster(np.array(fft_bottleneck[:, 2:50]), source_num)
        elif self.clustering_alg == "NMF":
            W, H, k_labels, _ = NMF_clustering.basis_exchange(W, H, np.array([source_num]), segment_width = 100, seq_nmf = False)
        elif self.clustering_alg == "seqNMF": 
            #W, H, cost, loading, power = seqnmf(np.array(fft_bottleneck[:, 3:]), K=source_num, L=2, Lambda=0.005)
            print(type(input))
            print(input.shape)
            print(input[:8,:16])
            W, H, cost, loading, power = seqnmf(input, K=source_num, L=2, Lambda=0.000001)
            k_labels = np.argmax(H, axis=0)
            #W, H, k_labels, _ = NMF_clustering.basis_exchange(W, H, np.array([source_num]), segment_width = 40, seq_nmf = True) 
           
        return W, H, k_labels


    def source_separation(self, input, phase, mean, std, filedir, filename):
        """
          main function for blind source separation.
          Argument:
              input: npArray. 
                  log power spectrum (lps).
              phase: npArray.
                  phase is used to inverse lps to wavform.
              mean: npArray.
                  mean value of lps.
              std: npArray. 
                  variance of lps.
              filedir: string. 
                  directory for separated sources.
              filename: string. 
                  separated sources name.
        """
        feature_dim = self.FFT_dict['frequency_bins'][1] - self.FFT_dict['frequency_bins'][0]

        if self.args.model_type == "DAE_C":
            x = np.reshape((input.T), (-1, 1, 1, int(self.FFT_dict['FFTSize']/2+1)))[:, :, :,self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
        else:
            x = input.T[:, self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1]]
        x = torch.tensor(x).float().cuda()
        sources = torch.unsqueeze(torch.zeros_like(x), 0)
        # Encode input
        if self.clustering_alg == "seqNMF_WEIGHTS":
            #weight_layer = self.model.encoder.conv_layers[-2].weight.data.cpu().numpy()
            #print(type(weight_layer))
            weight_layer = torch.nn.parameter.Parameter(self.model.encoder.conv_layers[-2].weight).detach().cpu().numpy()
            #weight_layer = np.array(weight_layer, dtype=np.float)
            sources = self.weight_nmf(x, weight_layer, sources)
        else:
            latent_code = self.model.encoder(x)
            print(latent_code.shape)
            # MFA analysis for identifying latent neurons 's label
            W, H, label = self.MFA(latent_code.cpu().detach().numpy())
            # Reconstruct input
            #latent_code_hat = helpers.reconstruct(W,H)
            #latent_code_hat = np.delete(latent_code_hat, np.s_[6451:],0)
            #print(latent_code_hat.shape)
            #latent_code_hat = torch.from_numpy(latent_code_hat).float().cuda()
            #sources[0] = self.model.decoder(latent_code_hat)
            sources[0] = self.model.decoder(latent_code)
            # Discriminate latent code for different sources.
            sources = torch.cat((sources, torch.unsqueeze(self.model.decoder(latent_code), 0)), 0)
            for source_idx in range(0, self.source_num):
                #y_s = self.freq_modulation(source_idx, label, latent_code_hat)
                #y_s = self.freq_modulation(source_idx, label, latent_code_hat)
                H_k = np.array(H)
                H_k[:,:] = 0
                for i in range (0, H.shape[1]):
                    if (H_k[source_idx,i] == np.argmax(H[:,i])):
                        H_k[source_idx,i] = H[source_idx,i]
                #W_k = W
                #W_k[:,:,:] = 0
                #W_k[:,source_idx,:] = W[:,source_idx,:]
                y_s = helpers.reconstruct(W,H_k)
                y_s = torch.from_numpy(y_s).float().cuda()
                sources = torch.cat((sources, torch.unsqueeze(self.model.decoder(y_s), 0)), 0)
        # should be same from here
        sources = torch.squeeze(sources).permute(0, 2, 1).detach().cpu().numpy()
        # Source separation
        #for source_idx in range(0, self.source_num+1):
        #    sources[source_idx, :, :] = np.sqrt(10**((sources[source_idx, :, :]*std[self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1], :])+mean[self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1], :]))

        # Inverse separated sources of log power spectrum to waveform.
        input = np.sqrt(10**(input*std+mean))

        for source_idx in range(0, self.source_num+1):
            Result = np.array(input)
            print(Result.shape)
            #if(self.wienner_mask==True):
            #    # Reconstruct original signal
            #    if source_idx == 0:
            #        Result[self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1], :] = sources[0, :, :]
            #    else:
            #        Result[self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1], :] =  2*(sources[source_idx, :, :]/(np.sum(sources[1:, :, :], axis = 0)+1e-8))*sources[0, :, :]
            #else:#Wienner_mask==False
            Result[self.FFT_dict['frequency_bins'][0]:self.FFT_dict['frequency_bins'][1], :] = np.array(sources[source_idx, :, :])
            R = np.multiply(Result, phase)
            result = librosa.istft(R, hop_length=self.FFT_dict['Hop_length'], win_length=self.FFT_dict['Win_length'], window=scipy.signal.hamming, center=True)
            result = np.int16(result*32768)
            if source_idx == 0:
                result_path = "{0}reconstruct/".format(filedir)
            else:
                result_path = "{0}source{1}/".format(filedir, source_idx)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            wav.write("{0}{1}.wav".format(result_path,  filename), self.FFT_dict['sr'], result)