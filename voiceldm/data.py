import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import scipy.signal as s
import librosa
import soundfile as sf
from .audio import raw_waveform_to_fbank, TacotronSTFT


class AudioDataset(Dataset):
    def __init__(self, args, df, df_noise, clap_processor):
        self.df = df
        self.df_noise = df_noise
        
        # self.paths = args.paths
        # self.noise_paths = args.noise_paths

        self.uncond_text_prob = args.uncond_text_prob
        self.add_noise_prob = args.add_noise_prob
        
        self.duration = 10
        self.target_length = int(self.duration * 102.4)
        self.stft = TacotronSTFT(
            filter_length=1024,
            hop_length=160,
            win_length=1024,
            n_mel_channels=64,
            sampling_rate=16000,
            mel_fmin=0,
            mel_fmax=8000,
        )

        self.clap_processor = clap_processor
        
    def pad_wav(self, wav, target_len, random_cut=False):
        n_channels, wav_len = wav.shape
        if n_channels == 2:
            wav = wav.mean(-2, keepdim=True)

        if wav_len > target_len:
            if random_cut:
                i = random.randint(0, wav_len - target_len)
                return wav[:, i:i+target_len]
            return wav[:, :target_len]
        elif wav_len < target_len:
            wav = F.pad(wav, (0, target_len-wav_len))
        return wav
    
    def reverb_rir(self, frames, rir):
        orig_frames_shape = frames.shape
        frames, filter = np.squeeze(frames), np.squeeze(rir)
        frames = s.convolve(frames, filter)
        actlev = np.max(np.abs(frames))
        if (actlev > 0.99):
            frames = (frames / actlev) * 0.98
        frames = frames[:orig_frames_shape[0]]
        return frames

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # row_rir = self.df_noise.iloc[index]
        # file_path = os.path.join(self.paths[row.data], row.file_path)
        file_path = os.path.join(row.file_path)

        noise_row = self.df_noise.iloc[random.randint(0, len(self.df_noise)-1)]
        rir = np.load(os.path.join(noise_row.file_path))
        frames,_ = librosa.load(file_path, sr=44100)
        frames = self.reverb_rir(frames,rir=rir)

        waveform, sr = torchaudio.load(file_path)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=32000)
        waveform = self.pad_wav(waveform, self.target_length * 160)

        noise = torch.from_numpy(frames).unsqueeze(0).float()       
        noise = torchaudio.functional.resample(noise, orig_freq=44100, new_freq=32000)
        noise = self.pad_wav(noise, self.target_length * 160)

        fbank, _, noise = raw_waveform_to_fbank(
            noise[0], 
            target_length=self.target_length, 
            fn_STFT=self.stft
        )
        
        # resample to 48k for clap
        wav_48k = torchaudio.functional.resample(noise, orig_freq=16000, new_freq=48000)
        clap_inputs = self.clap_processor(audios=wav_48k, return_tensors="pt", sampling_rate=48000)
        
        return fbank, noise, waveform, clap_inputs
        
    def __len__(self):
        return len(self.df)


class CollateFn:
    def __init__(self, text_processor):
        self.text_processor = text_processor

    def __call__(self, examples):
        fbank = torch.stack([example[0] for example in examples])
        waveform = torch.stack([example[1] for example in examples])
        clap_input_features = torch.cat([example[3].input_features for example in examples])
        clap_is_longer = torch.cat([example[3].is_longer for example in examples])
        
        text_tokens = torch.stack([example[2] for example in examples])


        return {
            "fbank": fbank, 
            "waveform": waveform, 
            "text_tokens": text_tokens,
            "clap_input_features": clap_input_features,
            "clap_is_longer": clap_is_longer,
        }