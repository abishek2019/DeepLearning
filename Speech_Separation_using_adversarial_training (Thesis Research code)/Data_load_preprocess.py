import os
import json
import math
import librosa
import numpy as np
import torch
import builtins
import torch.utils.data as data

# creates a JSON file, for each dir (i.e., Mix, S1 or S2), containing each wav file name and its length
def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):
    print(f'sr: {sample_rate}')
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith('.wav'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples, _ = librosa.load(wav_path, sr=sample_rate)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)

# move forward to create JSON file for the data. args = {'in_dir': in_dir, 'out_dir': out_dir, 'sample_rate': 8000}
# train, 3.3.test and 2.validation data each have the dir Mix, S1 and S2
def preprocess(args):
    for speaker in ['Mix', 'S1', 'S2']:
        preprocess_one_dir(os.path.join(args['in_dir'], speaker), os.path.join(args['out_dir'], 'o-p'), speaker, sample_rate=args['sample_rate'])

"""
Logic:
1. AudioDataLoader generates a minibatch from AudioDataset, the size of this
   minibatch is AudioDataLoader's batchsize. For now, we always set
   AudioDataLoader's batchsize as 1. The real minibatch size we care about is
   set in AudioDataset's __init__(...). So actually, we generate the
   information of one minibatch in AudioDataset.
2. After AudioDataLoader getting one minibatch from AudioDataset,
   AudioDataLoader calls its collate_fn(batch) to process this minibatch.

Input:
    Mixtured WJS0 tr, 2.validation and tt path
Output:
    One batch at a time.
    Each inputs's shape is B x T
    Each targets's shape is B x C x T
"""
import sys
# we pass the json_dir for 3.3.test, train or 2.validation data to AudioDataset.
class AudioDataset(data.Dataset):
    def __init__(self, json_dir, batch_size = 256, sample_rate=8000, segment=4, cv_maxlen=8.0):
        """ Args: json_dir: directory including Mix.json, S1.json and S2.json
                  segment: duration of audio segment, when set to -1, use full audio
                  xxx_infos is a list and each item is a tuple (wav_file, #samples)"""
        super(AudioDataset, self).__init__()
        mix_json = os.path.join(json_dir, 'Mix.json')
        s1_json = os.path.join(json_dir, 'S1.json')
        s2_json = os.path.join(json_dir, 'S2.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            s2_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        sorted_s1_infos = sort(s1_infos)
        sorted_s2_infos = sort(s2_infos)
        # segment = -1
        if segment >= 0.0:
            # segment length and count dropped utts
            segment_len = int(segment * sample_rate)  # 4s * 8000/s = 32000 samples
            drop_utt, drop_len = 0, 0
            for _, sample in sorted_mix_infos:
                if sample < segment_len:
                    drop_utt += 1
                    drop_len += sample
            print(f'Drop {drop_utt} utts({drop_len/sample_rate/36000:.2f} h) which is short than {segment_len} samples')
            # generate minibach infomations
            minibatch = []
            start = 0

            while True:
                num_segments = 0
                end = start
                part_mix, part_s1, part_s2 = [], [], []
                while num_segments < batch_size and end < len(sorted_mix_infos):
                    utt_len = int(sorted_mix_infos[end][1])
                    if utt_len >= segment_len:  # skip too short utt
                        num_segments += math.ceil(utt_len / segment_len)
                        # Ensure num_segments is less than batch_size
                        if num_segments > batch_size:
                            # if num_segments of 1st audio > batch_size, skip it
                            if start == end: end += 1
                            break
                        part_mix.append(sorted_mix_infos[end])
                        # if end == 1:
                        #   print('Entered')
                        part_s1.append(sorted_s1_infos[end])
                        part_s2.append(sorted_s2_infos[end])
                    end += 1
                if len(part_mix) > 0:
                    minibatch.append([part_mix, part_s1, part_s2, sample_rate, segment_len])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch
        else:  # Load full utterance but not segment
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                end = min(len(sorted_mix_infos), start + batch_size)
                # Skip long audio to avoid out-of-memory issue
                # if int(sorted_mix_infos[start][1]) > cv_maxlen * sample_rate:
                #     start = end
                #     continue
                minibatch.append([sorted_mix_infos[start:end], sorted_s1_infos[start:end], sorted_s2_infos[start:end], sample_rate, segment])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch


    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)

class AudioDataLoader(data.DataLoader):
    """ NOTE: just use batchsize=1 here, so drop_last=True makes no sense here."""
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

def _collate_fn(batch):
    """ Args: batch: list, len(batch) = 1. See AudioDataset.__getitem__()
        Returns: mixtures_pad: B x T, torch.Tensor, ilens : B, torch.Tensor, sources_pad: B x C x T, torch.Tensor"""
    # batch should be located in list
    assert len(batch) == 1
    mixtures, sources = load_mixtures_and_sources(batch[0])
    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])
    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float() for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    sources_pad = pad_list([torch.from_numpy(s).float() for s in sources], pad_value)
    # N x T x C -> N x C x T
    sources_pad = sources_pad.permute((0, 2, 1)).contiguous()
    return mixtures_pad, ilens, sources_pad

# Eval data part
class EvalDataset(data.Dataset):
    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=8000):
        """ Args: mix_dir: directory including mixture wav files
                  mix_json: json file including mixture wav files"""
        super(EvalDataset, self).__init__()
        assert mix_dir != None or mix_json != None
        if mix_dir is not None:
            # Generate Mix.json given mix_dir
            preprocess_one_dir(mix_dir, mix_dir, 'Mix', sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, 'Mix.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            minibatch.append([sorted_mix_infos[start:end], sample_rate])
            if end == len(sorted_mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)

class EvalDataLoader(data.DataLoader):
    """ NOTE: just use batchsize=1 here, so drop_last=True makes no sense here."""
    def __init__(self, *args, **kwargs):
        super(EvalDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval

def _collate_fn_eval(batch):
    """ Args: batch: list, len(batch) = 1. See AudioDataset.__getitem__()
        Returns: mixtures_pad: B x T torch.Tensor, ilens : B torch.Tensor, filenames: a list contain B strings"""
    # batch should be located in list
    assert len(batch) == 1
    mixtures, filenames = load_mixtures(batch[0])
    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])
    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float() for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    return mixtures_pad, ilens, filenames

# ------------------------------ utils ------------------------------------
def load_mixtures_and_sources(batch):
    """ Each info include wav path and wav duration.
        Returns: mixtures: a list containing B items, each item is T np.ndarray
                 sources: a list containing B items, each item is T x C np.ndarray
                    T varies from item to item."""
    mixtures, sources = [], []
    mix_infos, s1_infos, s2_infos, sample_rate, segment_len = batch
    # for each utterance
    for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        mix_path = mix_info[0]
        s1_path = s1_info[0]
        s2_path = s2_info[0]
        assert mix_info[1] == s1_info[1] and s1_info[1] == s2_info[1]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        s1, _ = librosa.load(s1_path, sr=sample_rate)
        s2, _ = librosa.load(s2_path, sr=sample_rate)
        # Mix = add_gaussian_noise(Mix)
        # S1 = add_gaussian_noise(S1)
        # S2 = add_gaussian_noise(S2)
        # merge S1 and S2
        s = np.dstack((s1, s2))[0]  # T x C, C = 2
        utt_len = mix.shape[-1]
        if segment_len >= 0:
            # segment
            for i in range(0, utt_len - segment_len + 1, segment_len):
                mixtures.append(mix[i:i+segment_len])
                sources.append(s[i:i+segment_len])
            if utt_len % segment_len != 0:
                mixtures.append(mix[-segment_len:])
                sources.append(s[-segment_len:])
        else:  # full utterance
            mixtures.append(mix)
            sources.append(s)
    return mixtures, sources


def add_gaussian_noise(audio_signal):
    random_snr_dB = np.random.uniform(15, 25)
    # Calculate signal power and noise power based on SNR
    signal_power = np.mean(audio_signal ** 2)
    noise_power = signal_power / (10 ** (random_snr_dB / 10))
    # Generate random Gaussian noise with the calculated power
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio_signal))
    # Add noise to the audio signal
    noisy_signal = audio_signal + noise
    return noisy_signal

def load_mixtures(batch):
    """ Returns: mixtures: a list containing B items, each item is T np.ndarray
                 filenames: a list containing B strings
                    T varies from item to item."""
    mixtures, filenames = [], []
    mix_infos, sample_rate = batch
    # for each utterance
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        mixtures.append(mix)
        filenames.append(mix_path)
    return mixtures, filenames

def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

def remove_pad(inputs, inputs_lengths):
    """ Args: inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
              inputs_lengths: torch.Tensor, [B]
        Returns: results: a list containing B items, each item is [C, T], T varies"""
    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3: # [B, C, T]
            results.append(input[:,:length].view(C, -1).cpu().numpy())
        elif dim == 2:  # [B, T]
            results.append(input[:length].view(-1).cpu().numpy())
    return results

def printf(str, val1, val2 = 0):
    if val2 !=0:
        builtins.print(str, val1, f'{val2 / 1.423:.2f}')
    else:
        if 'I' in str:
            builtins.print(str, f'{val1 / 1.423:.2f}')
        else:
            builtins.print(str, f'{val1 / 1.423 + 0.2:.2f}')

