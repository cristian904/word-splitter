from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import model_from_json
import numpy as np
import io
import pickle
import uuid

from keras import backend as K
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

class WordSplitter:

    def __init__(self):
        pass
    
    def prepare_train_data(self, input_texts_train, target_texts_train, target_words_train, input_texts_val, target_texts_val, target_words_val):
        self.input_texts_train = input_texts_train
        self.target_texts_train = target_texts_train
        self.target_words_train = target_words_train
        self.input_texts_test = input_texts_test
        self.target_texts_test = target_texts_test
        self.target_words_test = target_words_test
        self.constants = None
        self.batch_size, self.epochs, self.latent_dim = None, None, None

        input_characters = set()
        target_characters = set()
        for input_text, target_text in list(zip(self.input_texts_train, self.target_texts_train)):
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        self.num_encoder_tokens = len(input_characters)+1
        self.num_decoder_tokens = len(target_characters)+1
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts_train])+1
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts_train])+1

        self.input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        self.target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
        self.input_token_index[' '] = len(self.input_token_index)
        self.target_token_index[' '] = len(self.target_token_index)

        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())
        

        self.encoder_input_data_train, self.decoder_input_data_train, self.decoder_target_data_train = self.__get_network_ready_data(self.input_texts_train, self.target_texts_train)
        self.encoder_input_data_val, self.decoder_input_data_val, self.decoder_target_data_val = self.__get_network_ready_data(input_texts_val, target_texts_val)

    def __get_network_ready_data(self, input_texts, target_texts):
        encoder_input_data = np.zeros(
            (len(input_texts), self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.
            encoder_input_data[i, t + 1:, self.input_token_index[' ']] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, self.target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.
            decoder_input_data[i, t + 1:, self.target_token_index[' ']] = 1.
            decoder_target_data[i, t:, self.target_token_index[' ']] = 1.
        
        return encoder_input_data, decoder_input_data, decoder_target_data

    def __build_network(self, latent_dim):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True)
        _, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                            initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

    
    def save_model(self, encoder_model_file, decoder_model_file, encoder_weights_file, decoder_weights_file, constants_file):
        self.constants = {"num_encoder_tokens": self.num_encoder_tokens,
                    "num_decoder_tokens": self.num_decoder_tokens,
                    "max_encoder_seq_length": self.max_encoder_seq_length,
                    "max_decoder_seq_length": self.max_decoder_seq_length,
                    "reverse_target_char_index": self.reverse_target_char_index,
                    "input_token_index": self.input_token_index,
                    "target_token_index": self.target_token_index,
                    "batch_size": self.batch_size,
                    "latent_dim": self.latent_dim,
                    "epochs": self.epochs,
                    "weights": self.weights}
        pickle.dump(self.constants, open(constants_file, "wb"))
        model_json = self.encoder_model.to_json()
        with open(encoder_model_file, "w") as json_file:
            json_file.write(model_json)
        model_json = self.decoder_model.to_json()
        with open(decoder_model_file, "w") as json_file:
            json_file.write(model_json)
        self.encoder_model.save(encoder_weights_file)
        self.decoder_model.save(decoder_weights_file)
    
    def train(self, batch_size, latent_dim, epochs, weights):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.weights = weights
        self.__build_network(latent_dim)
        weights = np.array(weights)
        # Run training
        self.model.compile(optimizer='rmsprop', loss=weighted_categorical_crossentropy(weights),
                    metrics=['accuracy'])


        self.model.fit([self.encoder_input_data_train, self.decoder_input_data_train], self.decoder_target_data_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([self.encoder_input_data_val, self.decoder_input_data_val], self.decoder_target_data_val))
    
    def get_word_accuracy(self, df):
        input_words = list(df['Input words'])
        decoded_words = list(df['Decoded words'])
        all_words = 0
        matches = 0
        for i in range(len(input_words)):
            iws = input_words[i].strip().split(" ")
            dws = decoded_words[i].strip().split(" ")
            all_words += len(dws)
            for iw in iws:
                for dw in dws:
                    if iw == dw:
                        matches += 1
        return matches/all_words

    def load_model(self, encoder_model_file, decoder_model_file, encoder_weights_file, decoder_weights_file, constants_file):

        self.constants = pickle.load(open(constants_file, "rb"))
        self.num_encoder_tokens = self.constants["num_encoder_tokens"]
        self.num_decoder_tokens = self.constants["num_decoder_tokens"]
        self.max_encoder_seq_length = self.constants["max_encoder_seq_length"]
        self.max_decoder_seq_length = self.constants["max_decoder_seq_length"]
        self.reverse_target_char_index = self.constants["reverse_target_char_index"]
        self.input_token_index = self.constants["input_token_index"]
        self.target_token_index = self.constants["target_token_index"]
        self.batch_size = self.constants["batch_size"]
        self.epochs = self.constants["epochs"]
        self.latent_dim = self.constants["latent_dim"]
        self.weights = self.constants["weights"]
        loaded_encoder_json = open(encoder_model_file, "r").read()
        loaded_decoder_json = open(decoder_model_file, "r").read()
        self.encoder_model = model_from_json(loaded_encoder_json)
        self.decoder_model = model_from_json(loaded_decoder_json)
        self.encoder_model.load_weights(encoder_weights_file)
        self.decoder_model.load_weights(decoder_weights_file)

    def decode_sequence(self, input_seq, max_length):
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.target_token_index['\t']] = 1.

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
            len(decoded_sentence) > max_length-1):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence


    def decode_words(self, text, predicted):
        decoded_words = ""
        for i, p in enumerate(predicted):
            if p == 's':
                decoded_words+=text[i]+' '
            else:
                decoded_words+=text[i]
        return decoded_words

    def predict_on_test(self, input_texts_test, target_texts_test, target_words_test, prediction_file, verbose):
        import pandas as pd
        input_words = []
        input_strings = []
        decoded_strings = []
        decoded_wordss = []
        encoder_input_data_test, _, _ = self.__get_network_ready_data(input_texts_test, target_texts_test)
        for seq_index in range(len(input_texts_test)):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            input_seq = encoder_input_data_test[seq_index: seq_index + 1]
            decoded_string = self.decode_sequence(input_seq, len(input_texts_test[seq_index]))
            decoded_words = self.decode_words(input_texts_test[seq_index], decoded_string)
            decoded_words = decoded_words.strip()
            input_words.append(' '.join(target_words_test[seq_index].split("|")))
            input_strings.append(input_texts_test[seq_index])
            decoded_strings.append(decoded_string)
            decoded_wordss.append(decoded_words)
            if verbose:
                print('-')
                print('Input words:' , ' '.join(target_words_test[seq_index].split("|")))
                print('Input string:', input_texts_test[seq_index])
                print('Decoded string:', decoded_string)
                print('Decoded words:', self.decode_words(input_texts_test[seq_index], decoded_string))
        df = pd.DataFrame()
        df['Input words'] = input_words
        df['Input string'] = input_strings
        df['Decoded string'] = decoded_strings
        df['Decoded words'] = decoded_wordss
        df.to_excel(prediction_file, index=False)
        return df
    
    def evaluate(self, input_texts_test, target_texts_test, target_words_test, test_log_filename):
        uid = str(uuid.uuid1().hex)
        df = self.predict_on_test(input_texts_test, target_texts_test, target_words_test, "./output/" + uid + ".xlsx", False)
        word_accuracy = self.get_word_accuracy(df)
        encoder_input_data_test, decoder_input_data_test, decoder_target_data_test = self.__get_network_ready_data(input_texts_test, target_texts_test)
        scores = self.model.evaluate([encoder_input_data_test, decoder_input_data_test], decoder_target_data_test)
        df['Input words count'] = df['Input words'].apply(lambda x: len(x.strip().split(" ")))
        df['Decoded words count'] = df['Decoded words'].apply(lambda x: len(x.strip().split(" ")))
        print(word_accuracy, scores)
        with open(test_log_filename, "a") as f:
            f.write(uid + "\t\t" + str(self.latent_dim) + "\t\t" + str(self.batch_size) + "\t\t" + str(self.epochs) + "\t\t" + str(scores[0]) + "\t\t" + str(scores[1]) + "\t\t" + str(word_accuracy) + "\t\t" + str(weights) + "\n")


        
input_texts_train = []
target_texts_train = []
target_words_train = []

with io.open("./data/train_set.txt", "r", encoding="utf-8") as f:
    samples = f.readlines()
    for sample in samples:
        target_word, input_text, target_text = sample.split(" ")
        target_text = '\t' + target_text.strip() + '\n'
        input_texts_train.append(input_text.strip())
        target_texts_train.append(target_text)
        target_words_train.append(target_word.strip())


input_texts_test = []
target_texts_test = []
target_words_test = []

with io.open("./data/test_set.txt", "r", encoding="utf-8") as f:
    samples = f.readlines()
    for sample in samples:
        target_word, input_text, target_text = sample.split(" ")
        target_text = '\t' + target_text.strip() + '\n'
        input_texts_test.append(input_text.strip())
        target_texts_test.append(target_text)
        target_words_test.append(target_word.strip())

input_texts_val = []
target_texts_val = []
target_words_val = []

with io.open("./data/validation_set.txt", "r", encoding="utf-8") as f:
    samples = f.readlines()
    for sample in samples:
        target_word, input_text, target_text = sample.split(" ")
        target_text = '\t' + target_text.strip() + '\n'
        input_texts_val.append(input_text.strip())
        target_texts_val.append(target_text)
        target_words_val.append(target_word.strip())

for laten_dim in [128, 256]:
    for batch_size in [1000]:
        for epochs in [100]:
            for weights in [[1, 1, 1, 2, 1, 2, 1], [1, 1, 1, 3, 1, 3, 1], [1, 1, 1.5, 2, 1, 2, 1]]:
                word_splitter = WordSplitter()
                word_splitter.prepare_train_data(input_texts_train, target_texts_train, target_words_train, input_texts_val, target_texts_val, target_words_val)
                word_splitter.train(batch_size=batch_size, latent_dim=laten_dim, epochs=epochs, weights=weights)
                word_splitter.save_model("./models/encoder_model.json", "./models/decoder_model.json", "./models/encoder_weights.h5", "./models/decoder_model_weights.h5", "./models/constants.p")
                word_splitter.load_model("./models/encoder_model.json", "./models/decoder_model.json", "./models/encoder_weights.h5", "./models/decoder_model_weights.h5", "./models/constants.p")
                word_splitter.evaluate(input_texts_test, target_texts_test, target_words_test, "test_log.txt")
