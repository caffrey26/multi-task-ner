# -*- coding: utf-8 -*-
# Define dataset paths
import enum
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, TimeDistributed,  Bidirectional, concatenate, Dense, SpatialDropout1D
from keras.models import model_from_json


# enum to define dataset paths
class DataSet (enum.Enum):
  ATIS = 'atis.fold0.pkl'
  MIT_R = 'restauranttrain.bio'
  MIT_M_ENG = 'engtrain.bio'
  MIT_M_TRIVIA = 'trivia10k13train.bio'

sentences = [] # list of sentences from all the datasets, where each sentence is a list of words
tags = [] # list of tags associated with sentences, each tag is a list of tags to words
max_len = 75 # maximum sentence length
max_len_char = 30 # maximum word length


# convert from index to words, and vice-versa using dictionary
def convert_idx_word (dict, lst):
  return [[ dict[idx] for idx in sents] for sents in lst]

# return a flipped dictionary where values become keys
def flip_dict (dict):
  return {val:key for key, val in dict.items()}

# function to preprocess ATIS dataset
def preprocess_atis (path):
  with open(path, 'rb') as f:
    try:
        train_set, valid_set, test_set, dicts = pickle.load(f, encoding='latin1')
    except:
        train_set, valid_set, test_set, dicts = pickle.load(f)
  
  idx2words = flip_dict (dicts['words2idx'])
  idx2labels = flip_dict (dicts['labels2idx'])
  
  # convert sentence, and labels from list of indices to list of words
  train_sentences = convert_idx_word(idx2words, train_set[0])
  train_labels = convert_idx_word (idx2labels, train_set[2])
    
  # same procedure for validation sentences and labels
  valid_sentences = convert_idx_word (idx2words, valid_set[0])
  valid_labels = convert_idx_word (idx2labels, valid_set[2])
  
  ret_sentences = train_sentences + valid_sentences
  ret_labels = train_labels + valid_labels
  
  print ('Dataset {}: Number of sentences: {} Number of tags: {}'.
         format( DataSet(path).name,  
                 len(ret_sentences), 
                 len(ret_labels) 
               )
        )
  
  # return the merged sentences
  return (ret_sentences, ret_labels)

# function to remove digits from sentences, and replace them with DIGIT
# Example: '12' => 'DIGITDIGIT'; 'G1B1' => 'GDIGITBDIGIT'
def remove_digits (sents):
  for i, sent in enumerate(sents):
    for j, word in enumerate (sent):
      if word.isdigit():
        sents[i][j] = len(word)*"DIGIT"
      else:
        charInside = 0
        for char in word:
          if char.isdigit():
            charInside = 1
            break
        if charInside == 1:
          newWord = ""
          for char in word:
            if not char.isdigit():
              newWord += char
            else:
              newWord += "DIGIT"
          sents[i][j] = newWord
  return sents

  
# function to preprocess data in BIO format
def preprocess_bio (path):
  with open(path, 'r') as f:
    content = f.readlines()
    
  ret_sentences = []
  ret_labels = []

  sentence = []
  tag = []
  for line in content:
    if line != '\n':
      sentence.append(line.split()[1])
      tag.append(line.split()[0])
    else:
      ret_sentences.append(sentence)
      ret_labels.append(tag)
      sentence = []
      tag = []
  
  print ('Dataset {}: Number of sentences: {} Number of tags: {}'.
         format( DataSet(path).name,  
                 len(ret_sentences), 
                 len(ret_labels) 
               )
        )
  
  # check if sentences contain DIGITS, if it does replace them with DIGITS
  ret_sentences = remove_digits(ret_sentences)
  
  return (ret_sentences, ret_labels)  
   
  
# function to read through data sets, and append them to sentences, and tags
def preprocess_data():
  for dataset in DataSet:
    sents = ()
    # ATIS requires a different preprocessing
    if dataset == DataSet['ATIS']:
      sents = preprocess_atis(DataSet['ATIS'].value)
    else: 
      sents = preprocess_bio(dataset.value)
    if sents:
      sentences.extend (sents[0])
      tags.extend (sents[1])

def create_vocab():
  # words
  words_vocab = set()
  for sent in sentences:
    words_vocab.update(sent)

  words_vocab = list(words_vocab)
  n_words = len(words_vocab)
  print ('Number of unique words: ', n_words)
  
  # tags
  tags_vocab = set()
  for tag in tags:
    tags_vocab.update(tag)

  tags_vocab = list(tags_vocab)
  n_tags = len(tags_vocab)
  print ('Number of unique tags: ', n_tags)
  
  # characters
  chars_vocab = set([w_i for w in words_vocab for w_i in w])
  n_chars = len(chars_vocab)
  print ('Number of unique characters: ', n_chars)

  return words_vocab, n_words, tags_vocab, n_tags, chars_vocab, n_chars

# function to convert, and pad sentences as a list of indices of words, and chars,
# also converting the tags to a list of indices
def prepare_data(words_vocab, tags_vocab, chars_vocab):
  # create dictionaries
  word2idx = {w: i + 2 for i, w in enumerate(words_vocab)} # first 2 indices saved for special words
  word2idx["UNK"] = 1 # unknown word
  word2idx["PAD"] = 0 # PAD-ding word  for sentences

  idx2word = {i: w for w, i in word2idx.items()}

  tag2idx = {t: i + 1 for i, t in enumerate(tags_vocab)}
  tag2idx["PAD"] = 0 # PAD-ded tag for the PADDED words

  idx2tag = {i: w for w, i in tag2idx.items()}
  
  char2idx = {c: i + 2 for i, c in enumerate(chars_vocab)}
  char2idx["UNK"] = 1 # unknown new char index
  char2idx["PAD"] = 0 # padded char index
  

  # change the sentence from a word structure to integer index structure
  X_word = [[word2idx[w] for w in s] for s in sentences]

  # pad each sentence to form same length
  X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')
  
  # represent each sentence as a sequence of words formed as a concatenation of char indices
  X_char = []
  for sentence in sentences:
      sent_seq = []
      for i in range(max_len):
          word_seq = []
          for j in range(max_len_char):
              try:
                  word_seq.append(char2idx.get(sentence[i][j]))
              except:
                  word_seq.append(char2idx.get("PAD"))
          sent_seq.append(word_seq)
      X_char.append(np.array(sent_seq))
      
  # change from tag to index
  y = [[tag2idx[tag] for tag in sen_tags] for sen_tags in tags]
  y = pad_sequences(maxlen=max_len, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')
  
  print('Data prepared! Ready to feed into the monster.')
  return X_word, X_char, y

def prepare_model(n_words, n_tags, n_chars):
  # input and embedding for words; each word will become a 20 dimensional word
  word_in = Input(shape=(max_len,))
  emb_word = Embedding(input_dim=n_words + 2, output_dim=20,
                       input_length=max_len, mask_zero=True)(word_in)

  # input and embeddings for characters
  char_in = Input(shape=(max_len, max_len_char,)) # each sentence contains max_len words, where each word contains max_len_char characters
  emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                             input_length=max_len_char, mask_zero=True))(char_in)

  # character LSTM to get word encodings by characters
  char_enc = TimeDistributed(Bidirectional(LSTM(units=20, return_sequences=False,
                                  recurrent_dropout=0.5)))(emb_char)

  # main LSTM
  x = concatenate([emb_word, char_enc]) # each word embedding is a combination of embeddings derived at word, and char levels.
  x = SpatialDropout1D(0.3)(x)
  main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                                 recurrent_dropout=0.6))(x)

  out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(main_lstm)

  model = Model([word_in, char_in], out)

  model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

  model.summary()
  
  return model
  
# ******************CODE BEGINS FROM HERE********************

# get the data in sentences, and tags format      
preprocess_data()
# print (sentences[4978], tags[4978])
print ('Total number of sentences: {} Total number of tags: {}\n'.format(len(sentences), len(tags)))

# create the vocabulary, and dictionaries now 
words_vocab, n_words, tags_vocab, n_tags, chars_vocab, n_chars = create_vocab()

# prepare the input for model
X_word, X_char, y = prepare_data(words_vocab, tags_vocab, chars_vocab)

# prepare the model
model = prepare_model(n_words, n_tags, n_chars)

# fit the model 
history = model.fit([X_word,
                     np.array(X_char).reshape((len(X_char), max_len, max_len_char))],
                    np.array(y).reshape(len(y), max_len, 1),
                    batch_size=32, epochs=10, validation_split=0.1, verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")

# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

