import glob
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram

# Extracting features from the .wav audio files of the training data set and test data set


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name, duration=5.0)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

# Go through each file for training and testing data set using the glob package
# glob.glob() matches the file names with the parameter given and extracts it


def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav"):
    features, labels = np.empty((0, 193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            except Exception as e:
                print("Error encountered while parsing file:", fn)
                continue
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_features])
            # print(fn[20:len(fn)].split('-')[2])
            # print (fn)
            lbl=int(fn[20:len(fn)].split('-')[2])-1
            labels = np.append(labels, lbl)
            # print(fn[20:len(fn)].split('-')[2])
            # print(int(fn[20:len(fn)].split('-')[2]))
    return np.array(features), np.array(labels, dtype=np.int)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


parent_dir = 'Sound-data'
tr_sub_dirs = ["Actor_01", "Actor_02", "Actor_03", "Actor_04", "Actor_05", "Actor_06", "Actor_07","Actor_08","Actor_09",
               "Actor_10", "Actor_11"]
ts_sub_dirs = ["Actor_12"]

# tr_sub_dirs = ["fold1", "fold2"]
# ts_sub_dirs = ["fold3"]

tr_features, tr_labels = parse_audio_files(parent_dir, tr_sub_dirs)
ts_features, ts_labels = parse_audio_files(parent_dir, ts_sub_dirs)

print(tr_labels)

tr_labels = one_hot_encode(tr_labels)
ts_labels = one_hot_encode(ts_labels)


# Define the training model and Run the session

training_epochs = 5000
n_dim = tr_features.shape[1]
n_classes = 8
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1/np.sqrt(n_dim)
learning_rate = 0.00001

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=0))
y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)


init = tf.global_variables_initializer()
cost_function = -tf.reduce_sum(Y*tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1], dtype=float)
y_true, y_prediction = None, None

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        _, cost = sess.run([optimizer, cost_function], feed_dict={X: tr_features, Y: tr_labels})
        cost_history = np.append(cost_history, cost)

    y_prediction = sess.run(tf.argmax(y_, 1), feed_dict={X: ts_features})
    y_true = sess.run(tf.argmax(ts_labels, 1))
    print("Test accuracy: ", round(sess.run(accuracy, feed_dict={X: ts_features, Y: ts_labels}), 3))










