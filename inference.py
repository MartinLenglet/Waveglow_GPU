# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
from scipy.io import wavfile
from scipy.io.wavfile import write
import torch
from mel2samp import files_to_list, MAX_WAV_VALUE
from denoiser import Denoiser

import numpy as np
import noisereduce as nr

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def main(mel_files, waveglow_path, sigma, output_dir, sampling_rate, is_fp16,
         denoiser_strength, factor_interp=1, gain=0, negative_gain=0):
    
    tic()

    mel_files = files_to_list(mel_files)
    waveglow = torch.load(waveglow_path)['model']
    num_param = get_param_num(waveglow)
    print("Number of Waveglow Parameters:", num_param)

    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    if is_fp16:
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()

    toc()

    for i, file_path in enumerate(mel_files):
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        if True:
            # Processing for generic mel files
            shape = tuple(np.fromfile(file_path, count = 2, dtype = np.int32))
            mel = np.memmap(file_path,offset=8,dtype=np.float32,shape=shape)
            # mel = mel[2000:3000,:]
            mel = mel.transpose() + gain - negative_gain
            # print(type(mel[0,0]))

            mel = torch.from_numpy(mel)
            size_interp = round(mel.size(1)*factor_interp)
            mel_interp = np.zeros((mel.size(0), size_interp))
            for i in range(0,mel.size(0)):
                mel_interp[i] = np.interp(np.linspace(0, 1, size_interp), np.linspace(0, 1, mel.size(1)), mel[i])
            # mel_interp = mel_interp.astype(float)
            # mel = torch.from_numpy(mel)
            mel = torch.from_numpy(mel_interp)
            mel = mel.float()

            if generate_stretching:
                ref_file_path = os.path.join(ref_folder_stretching, os.path.basename(file_path))
                ref_shape = tuple(np.fromfile(ref_file_path, count = 2, dtype = np.int32))
                ref_mel = np.memmap(ref_file_path,offset=8,dtype=np.float32,shape=ref_shape)
                ref_mel = ref_mel.transpose() + gain - negative_gain

                ref_mel = torch.from_numpy(ref_mel)
                ref_size_interp = mel.size(1) # mel.size(1) as been updated previosuly by factor_inter
                ref_mel_interp = np.zeros((ref_mel.size(0), ref_size_interp))
                for i in range(0, ref_mel.size(0)):
                    ref_mel_interp[i] = np.interp(np.linspace(0, 1, ref_size_interp), np.linspace(0, 1, ref_mel.size(1)), ref_mel[i])
                ref_mel = torch.from_numpy(ref_mel_interp)
                ref_mel = ref_mel.float()

        else:
            # mel = torch.load(file_path)
            mel = torch.from_numpy(np.load(file_path).transpose())

        # print(mel)
        # print(len(mel))
        # print(len(mel[0]))

        mel = torch.autograd.Variable(mel.cuda())
        mel = torch.unsqueeze(mel, 0)
        mel = mel.half() if is_fp16 else mel
        with torch.no_grad():
            audio = waveglow.infer(mel, sigma=sigma)
            if denoiser_strength > 0:
                audio = denoiser(audio, denoiser_strength)
            audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        audio_path = os.path.join(
            output_dir, "{}.wav".format(file_name))
        write(audio_path, sampling_rate, audio)
        print(audio_path)

        if generate_stretching:
            ref_mel = torch.autograd.Variable(ref_mel.cuda())
            ref_mel = torch.unsqueeze(ref_mel, 0)
            ref_mel = ref_mel.half() if is_fp16 else ref_mel
            with torch.no_grad():
                audio = waveglow.infer(ref_mel, sigma=sigma)
                if denoiser_strength > 0:
                    audio = denoiser(audio, denoiser_strength)
                audio = audio * MAX_WAV_VALUE
            audio = audio.squeeze()
            audio = audio.cpu().numpy()
            audio = audio.astype('int16')
            ref_audio_path = os.path.join(
                output_dir, "{}_stretch.wav".format(file_name))
            write(ref_audio_path, sampling_rate, audio)
            print(ref_audio_path)

        if post_denoiser:
            # Denoising
            rate, data = wavfile.read(audio_path)
            # perform noise reduction
            reduced_noise = nr.reduce_noise(
                y=data,
                sr=rate,
                prop_decrease=0.7,
                stationary=True,
                n_fft=512,
                n_std_thresh_stationary=1.5,
                chunk_size=600000,
                # freq_mask_smooth_hz=5000
            )
            wavfile.write(audio_path, rate, reduced_noise)

            if generate_stretching:
                # Denoising
                rate, data = wavfile.read(ref_audio_path)
                # perform noise reduction
                reduced_noise = nr.reduce_noise(
                    y=data,
                    sr=rate,
                    prop_decrease=0.7,
                    stationary=True,
                    n_fft=512,
                    n_std_thresh_stationary=1.5,
                    chunk_size=600000,
                    # freq_mask_smooth_hz=5000
                )
                wavfile.write(ref_audio_path, rate, reduced_noise)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')
    parser.add_argument("-sf", "--speed_factor", default=1, type=float,
                        help='Add a speed ratio to the synthesis')
    parser.add_argument("-g", "--gain", default=0, type=float,
                        help='Add a gain to mel-spectro (in dB)')
    parser.add_argument("-ng", "--negative_gain", default=0, type=float,
                        help='Add a negative gain to mel-spectro (in dB)')
    parser.add_argument("--denoiser", required=False, action="store_true",
                        help='Post-Processing Denoiser')
    parser.add_argument("--generate_stretching", required=False, action="store_true",
                        help='Generate equivalent length syntheses from reference spectrograms given in --ref_folder_stretching')
    parser.add_argument("--ref_folder_stretching", required=False, default='../../FastSpeech2-master/output/audio/_NEB_Fastspeech2_120000_phonPred_predictor_embeddings_test_ortho_emb_by_layer_mean_distrib_calib',
                        help='Reference Utterances to generate stretching')
    #parser.add_argument("--ref_folder_stretching", required=False, default='../_syn/_syn_predictor_ablation/_out_NEB_RF2_prenet128_phonPred_predictor_noEmbeddings_test_ortho_emb_by_layer_mean_distrib_calib',
    #                    help='Reference Utterances to generate stretching')
    args = parser.parse_args()

    global post_denoiser
    post_denoiser = args.denoiser
    global generate_stretching
    generate_stretching = args.generate_stretching
    global ref_folder_stretching
    ref_folder_stretching = args.ref_folder_stretching

    main(args.filelist_path, args.waveglow_path, args.sigma, args.output_dir,
         args.sampling_rate, args.is_fp16, args.denoiser_strength, args.speed_factor, args.gain, args.negative_gain)
