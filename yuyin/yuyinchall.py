from collections import Counter
from yuyinutils import spare_tuple_to_texts_ch, ndarray_to_text_ch
from yuyinutils import get_audio_and_transcriptch, pad_sequences
from yuyinutils import spare_tuple_from, get_wavs_labels
import tensorflow as tf

wav_path = ''
label_file = ''
wav_files, labels = get_wavs_labels(wav_path, label_file)

all_words = []
for label in labels:
    all_words += [word for word in label]
counter = Counter(all_words)
words = sorted(counter)
words_size = len(words)
word_num_map = dict(zip(words, range(words_size)))

n_input = 26
n_context = 9
batch_size = 8


def next_batch(labels, start_idx=0, batch_size=1, wav_files=wav_files):
    filesize = len(labels)
    end_idx = min(filesize, start_idx+batch_size)
    idx_list = range(start_idx, end_idx)
    txt_labels = [labels[i] for i in idx_list]
    wav_files = [wav_files[i] for i in idx_list]
    (source, audio_len, target, transcript_len) = get_audio_and_transcriptch(
        None, wav_files, n_input, n_context, word_num_map, txt_labels)

    start_idx += batch_size
    if start_idx >= filesize:
        start_idx = -1
    source, source_lengths = pad_sequences(source)
    sparse_labels = spare_tuple_from(target)
    return start_idx, source, source_lengths, sparse_labels


next_idx, source, source_len, sparse_lab = next_batch(labels, 0, batch_size)
t = spare_tuple_to_texts_ch(sparse_lab, words)
input_tensor = tf.placeholder(
    tf.float32, [None, None, n_input+(2*n_input*n_context)], name='input')
target = tf.sparse_placeholder(tf.int32, name='targets')
seq_length = tf.placeholder(tf.int32, [None], name='seq_length')
keep_dropout = tf.placeholder(tf.float32)
logits = BiRNN_model(input_tensor, tf.to_int64(seq_length),
                     n_input, n_context, words_size+1, keep_dropout)

b_stddev = 0.046875
h_stddev = 0.046875

n_hidden = 1024
n_hidden_1 = 1024
n_hidden_2 = 1024
n_hidden_5 = 1024
n_hidden_3 = 2*1024
n_cell_dim = 1024

keep_dropout_rate = 0.95
relu_clip = 20


def variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def BiRNN_model(batch_x, seq_length, n_input, n_context, n_character, keep_dropout):
    batch_x_shape = tf.shape(batch_x)
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    batch_x = tf.reshape(batch_x, [-1, n_input+2*n_input*n_context])
    with tf.name_scope('fc1'):
        b1 = variable_on_cpu(
            'b1', [n_hidden_1], tf.random_normal_initializer(stddev=b_stddev))
        h1 = variable_on_cpu('h1', [n_input+2*n_input*n_context,
                                    n_hidden_1], tf.random_normal_initializer(stddev=h_stddev))
        layer_1 = tf.minimum(tf.nn.relu(
            tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
        layer_1 = tf.nn.dropout(layer_1, keep_dropout)
    with tf.name_scope('fc2'):
        b2 = variable_on_cpu(
            'b2', [n_hidden_2], tf.random_normal_initializer(stddev=b_stddev))
