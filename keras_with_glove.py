
# For installing glove on Centos machine following should be executed 
# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip /content/glove.6B.zip

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.metrics import classification_report
from keras.engine.input_layer import Input
from keras.models import Model


def read_glove(glove_path):
    '''

    :return:
    '''
    embeddings_index = dict()
    f = open(glove_path, encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def mat_to_label(df_col, np_array):
    '''
    
    '''
    le = LabelEncoder()
    le.fit(df_col)
    return le.inverse_transform(np_array)


#Generating submission csv by adding pred_labels columns to test with required column_name
def generate_pred_csv(testset, pred_labels_df, column_name, csv_name):
    """
    Converting pred_labels to dataframe
    :param testset:
    :param pred_labels_df:
    :param column_name:
    :param csv_name:
    :return: submission_csv
    """ 
    testset[column_name] = pred_labels_df
    return testset.to_csv(csv_name, index = False)
    
    
    
    

if __name__=='__main__':
  # For executing this code first code file weighted_approach.py should be executed, so as to get train and test
  # Using vector embeddings from GLOVE and Keras
  train['ingredients_flat'] = list_to_seq(train['ingredients'])
  val['ingredients_flat'] = list_to_seq(val['ingredients'])
  test['ingredients_flat'] = list_to_seq(test['ingredients'])
  train_val_combined = train.append(val)
  train_val_test_combined = train_val_combined.append(test)



  #
  le = LabelEncoder()
  train_labels = label_to_mat(train['cuisine'])
  val_labels = label_to_mat(val['cuisine'])

  #
  train_docs = train['ingredients_flat'].values
  val_docs = val['ingredients_flat'].values
  test_docs = test['ingredients_flat'].values
  combined_docs = train_val_test_combined['ingredients_flat'].values

  #
  t = Tokenizer()
  t.fit_on_texts(texts = combined_docs)
  vocab_size = len(t.word_index) + 1
  print(vocab_size)

  #
  encoded_docs = t.texts_to_sequences(train_docs)
  encoded_val_docs = t.texts_to_sequences(val_docs)
  encoded_test_docs = t.texts_to_sequences(test_docs)


  #
  max_length = 40
  padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
  padded_val_docs = pad_sequences(encoded_val_docs, maxlen=max_length, padding='post')
  padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_length, padding='post')


  # load the whole embedding into memory
  embeddings_index = read_glove('/content/glove.6B.100d.txt')


  #
  vocab = pd.DataFrame.from_dict(t.word_index, orient="index")
  vocab.drop([0], axis=1).reset_index().rename(columns={"index": "word"})


  # create a weight matrix for words in training docs
  embeddings_matrix = np.zeros((vocab_size, 100))
  for word, i in t.word_index.items():
      vector = embeddings_index.get(word)
      if vector is not None:
          embeddings_matrix[i] = vector
  print(embeddings_matrix.shape)





  # fix random seed for reproducibility
  seed = 42
  np.random.seed(seed)


  # define the model

  model = Sequential()

  model.add(Embedding(vocab_size, 100, weights = [embeddings_matrix], input_length=40, trainable=False))
  model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Flatten())
  model.add(Dense(250, activation='relu'))
  model.add(Dense(20, activation='sigmoid'))
  # compile the model
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
  # summarize the model
  print(model.summary())
  # fit the model
  model.fit(padded_docs, train_labels, epochs= 20, verbose=0)
  print(padded_val_docs.shape)
  print(val_labels.shape)
  scores = model.evaluate(padded_val_docs, val_labels, verbose=0)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


