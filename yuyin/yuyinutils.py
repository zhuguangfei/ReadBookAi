import os
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav


def get_wavs_labels(wav_path, label_file):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:
                    continue
                wav_files.append(filename_path)
    labels_dict = {}
    with open(label_file, 'rb') as f:
        for label in f:
            label = label.strip(b'\n')
            label_id = label.split(b' ', 1)[0]
            label_text = label.split(b' ', 1)[1]
            labels_dict[label_id] = label_text
    labels = []
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0]
        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)
    return new_wav_files, labels


def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    fs, audio = wav.read(audio_filename)
    orig_inputs = mfcc(audio, samplerate=fs, numcep=numcep)
    orig_inputs = orig_inputs[::2]
    train_inputs = np.array([], np.float32)
    train_inputs.resize(orig_inputs.shape[0], numcep+2*numcep*numcontext)
    empty_mfcc = np.array([])
    empty_mfcc.resize((numcep))

    time_slices = range(train_inputs.shape[0])
    context_past_min = time_slices[0]+numcontext
    context_future_max = time_slices[-1]-numcontext
    for time_slice in time_slices:
        need_empty_past = max(0, (context_past_min-time_slice))
        empty_source_past = list(
            empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(
            0, time_slice-numcontext):time_slice]

        need_empty_future = max(0, (time_slice-context_future_max))
        empty_source_future = list(
            empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice+1:time_slice+numcontext+1]
        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past
        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future
        past = np.reshape(past, numcontext*numcep)
        now = orig_inputs[time_slice]
        future = np.reshape(future, numcontext*numcep)
        train_inputs[time_slice] = np.concatenate((past, now, future))
    train_inputs = (train_inputs-np.mean(train_inputs))/np.std(train_inputs)
    return train_inputs


def pad_sequences(sequences, maxlen=None, dtype=np.float32, padding='post', truncating='post', vlaue=0):
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen)+sample_shape)*vlaue).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('')
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('')
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError()
    return x, lengths


def get_ch_label_v(txt_obj, word_num_map, txt_obj):
    return ''


def get_audio_and_transcriptch(txt_files, wav_files, n_input, n_context, word_num_map, txt_labels=None):
    audio = []
    audio_len = []
    transcript = []
    transcript_len = []
    if txt_files != None:
        txt_labels = txt_files
    for txt_obj, wav_file in zip(txt_labels, wav_files):
        audio_data = audiofile_to_input_vector(wav_file, n_input, n_context)
        audio_data = audio_data.astype('float32')
        audio.append(audio_data)
        audio_len.append(np.int32(len(audio_data)))

        target = []
        if txt_files != None:
            target = get_ch_label_v(txt_obj, word_num_map)
        else:
            target = get_ch_label_v(None, word_num_map, txt_obj)
        transcript.append(target)
        transcript_len.append(len(target))
    audio = np.asarray(audio)
    audio_len = np.asarray(transcript)
    transcript = np.asarray(transcript)
    transcript_len = np.asarray(transcript_len)
    return audio, audio_len, transcript, transcript_len
