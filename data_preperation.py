# torchaudio provides audio and music analysis on Python
# !pip install torchaudio librosa boto3  # Run only once
# !pip install wavio  # Run only once
# !pip install tensorflow  # Run only once
# Kapre implements time-frequency conversions, normalisation,  and data augmentation as Keras layers
# !pip install kapre  # Run only once

# Notes:
#  Data set : https://www.kaggle.com/vinayshanbhag/bird-song-data-set?select=bird_songs_metadata.csv
# ??  https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
# (Audio Preprocessing Layers for a Quick Implementation ofDeep Neural Network Models with Keras) chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://arxiv.org/pdf/1706.05781.pdf
#  https://keras.io/about/
#  https://pysoundfile.readthedocs.io/en/latest/
# KAPRE library : https://github.com/keunwoochoi/kapre/blob/master/kapre/time_frequency.py
# wandg report https://wandb.ai/skycladai/bird-classification/reports/Bird-Classification-Report--Vmlldzo5ODg1NjI


import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile as sf
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono
from tensorflow.python.keras.backend import dtype
from tqdm import tqdm
import wavio


def envelope(y, rate, threshold):
    # removes the sound without signal and keeps the sound that includes value(data)
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def downsample_mono(path, sr):
    # wavio library does not read the floating point (24 bit) files
    # so we'll use the soundfile library to make the convertion
    data, samplerate = sf.read(path)
    # Changes the format from 24 bit to 16 bit
    sf.write(path, data, samplerate, subtype='PCM_16')

    obj = wavio.read(file=path)
    wav = obj.data.astype(np.float32, order='F')
    rate = obj.rate

    try:
        channel = wav.shape[1]
        if channel == 2:
            wav = to_mono(wav.T)
        elif channel == 1:
            wav = to_mono(wav.reshape(-1))
    except IndexError:
        wav = to_mono(wav.reshape(-1))
        pass
    except Exception as exc:
        raise exc
    wav = resample(wav, rate, sr)
    wav = wav.astype(np.int16)
    return sr, wav


def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir.split(
        '.')[0], fn+'_{}.wav'.format(str(ix)))
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def split_wavs(args):
    src_root = args.src_root
    dst_root = args.dst_root
    dt = args.delta_time

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    dirs = os.listdir(src_root)
    check_dir(dst_root)
    classes = os.listdir(src_root)
    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
        for fn in tqdm(os.listdir(src_dir)):
            src_fn = os.path.join(src_dir, fn)
            rate, wav = downsample_mono(src_fn, args.sr)
            mask, y_mean = envelope(wav, rate, threshold=args.threshold)
            wav = wav[mask]
            delta_sample = int(dt*rate)

            # cleaned audio is less than a single sample
            # pading with zeros to delta_sample size
            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                sample[:wav.shape[0]] = wav
                save_sample(sample, rate, target_dir, fn, 0)
            # step through audio and save every delta_sample
            # discard the ending audio if it is too short
            else:
                trunc = wav.shape[0] % delta_sample
                for cnt, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                    start = int(i)
                    stop = int(i + delta_sample)
                    sample = wav[start:stop]
                    save_sample(sample, rate, target_dir, fn, cnt)


def test_threshold(args):
    src_root = args.src_root
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_path = [x for x in wav_paths if args.fn in x]
    # print(wav_path[0])
    if len(wav_path) != 1:
        print('audio file not found for sub-string: {}'.format(args.fn))
        return
    rate, wav = downsample_mono(wav_path[0], args.sr)
    mask, env = envelope(wav, rate, threshold=args.threshold)
    plt.style.use('ggplot')
    plt.title('Signal Envelope, Threshold = {}'.format(str(args.threshold)))
    plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
    plt.plot(wav[mask], color='c', label='keep')
    plt.plot(env, color='m', label='envelope')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


def wav_assortment(dst_root):
    audio_csv = pd.read_csv('bird_songs_metadata.csv')

    all_ids = audio_csv['id']
    # arrange the wav files inside a folder with the name of species
    for idx, id in all_ids.iteritems():
        title = audio_csv['species'][idx]
        destination = dst_root+"/"+title
        check_dir(destination)
        for i in range(0, 20):
            path = dst_root+"/"+str(id)+"-"+str(i) + ".wav"
            if os.path.isfile(path):
                os.replace(path, destination+"/"+str(id)+"-"+str(i) + ".wav")
    # Delete the unsorted files
    filtered_files = os.listdir(dst_root)
    for file in filtered_files:
        if os.path.isfile(file):
            path_to_file = os.path.join(dst_root, file)
            os.remove(path_to_file)


if __name__ == '__main__':  # (Run it 3 times)
    # wav_assortment("wavfiles") # Run only once. comment out after use (1st run)
    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_root', type=str, default='wavfiles',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default='clean',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='rate to downsample audio. 16000 is the number of samples taken per second')

    parser.add_argument('--fn', type=str, default='12577-0',
                        help='file to plot over time to check magnitude')
    parser.add_argument('--threshold', type=str, default=50,
                        help='threshold magnitude for np.int16 dtype (with threshold to 50 creates more accurate filter)')
    args, _ = parser.parse_known_args()

    # comment out after use
    # test_threshold(args) # (2nd run)
    split_wavs(args)  # (3rd run)
