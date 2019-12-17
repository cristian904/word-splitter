from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import model_from_json
import numpy as np
import io
import pickle

batch_size = 100  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

class WordSplitter:

    def __init__(self, input_texts_train, target_texts_train, target_words_train, input_texts_test, target_texts_test, target_words_test):
        self.input_texts_train = input_texts_train
        self.target_texts_train = target_texts_train
        self.target_words_train = target_words_train
        self.input_texts_test = input_texts_test
        self.target_texts_test = target_texts_test
        self.target_words_test = target_words_test
        self.encoder_input_data_test = None
        self.decoder_input_data_test = None
        self.decoder_target_data_test = None
    
    def prepare_data(self):
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
        pickle.dump(self.reverse_target_char_index, open("reverse_target_char_index.p", "wb"))

        self.encoder_input_data_train, self.decoder_input_data_train, self.decoder_target_data_train = self.get_network_ready_data(self.input_texts_train, self.target_texts_train)
        self.encoder_input_data_test, self.decoder_input_data_test, self.decoder_target_data_test = self.get_network_ready_data(self.input_texts_test, self.target_texts_test)

    def get_network_ready_data(self, input_texts, target_texts):
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

    def __build_network(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
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

    
    def save_model(self, encoder_model_file, decoder_model_file, encoder_weights_file, decoder_weights_file):
        model_json = self.encoder_model.to_json()
        with open(encoder_model_file, "w") as json_file:
            json_file.write(model_json)
        model_json = self.decoder_model.to_json()
        with open(decoder_model_file, "w") as json_file:
            json_file.write(model_json)
        self.encoder_model.save(encoder_weights_file)
        self.decoder_model.save(decoder_weights_file)
    
    def train(self, ):

        self.__build_network()
        
        # Run training
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                    metrics=['accuracy'])


        self.model.fit([self.encoder_input_data_train, self.decoder_input_data_train], self.decoder_target_data_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([self.encoder_input_data_test, self.decoder_input_data_test], self.decoder_target_data_test))


    def load_model(self, encoder_model_file, decoder_model_file, encoder_weights_file, decoder_weights_file, reverse_target_char_index_file):

        self.reverse_target_char_index = pickle.load(open(reverse_target_char_index_file, "rb"))
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
            if p == 'i':
                decoded_words+=' '+text[i]
            elif p == 's':
                decoded_words+=text[i]+' '
            else:
                decoded_words+=text[i]
        return decoded_words

    def predict_on_test(self):
        for seq_index in range(len(self.input_texts_train)):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            input_seq = self.encoder_input_data_test[seq_index: seq_index + 1]
            decoded_string = self.decode_sequence(input_seq, len(self.input_texts_test[seq_index]))
            print('-')
            print('Input words:' , ' '.join(self.target_words_test[seq_index].split("|")))
            print('Input string:', self.input_texts_test[seq_index])
            print('Decoded string:', decoded_string)
            print('Decoded words:', self.decode_words(self.input_texts_test[seq_index], decoded_string))


        
input_texts_train = []
target_texts_train = []
target_words_train = []

with io.open("./data/train_set.txt", "r", encoding="utf-8") as f:
    samples = f.readlines()
    for sample in samples:
        target_word, input_text, target_text = sample.split(" ")
        target_text = '\t' + target_text + '\n'
        input_texts_train.append(input_text)
        target_texts_train.append(target_text)
        target_words_train.append(target_word)


input_texts_test = []
target_texts_test = []
target_words_test = []

with io.open("./data/test_set.txt", "r", encoding="utf-8") as f:
    samples = f.readlines()
    for sample in samples:
        target_word, input_text, target_text = sample.split(" ")
        target_text = '\t' + target_text + '\n'
        input_texts_test.append(input_text)
        target_texts_test.append(target_text)
        target_words_test.append(target_word)


word_splitter = WordSplitter(input_texts_train, target_texts_train, target_words_train, input_texts_test, target_texts_test, target_words_test)
word_splitter.prepare_data()
word_splitter.train()
word_splitter.save_model("./models/encoder_model.json", "./models/decoder_model.json", "./models/encoder_weights.h5", "./models/decoder_model_weights.h5")
# word_splitter.load_model("encoder_model.json", "decoder_model.json", "encoder_weights.h5", "decoder_model_weights.h5", "reverse_target_char_index.p")
# word_splitter.predict_on_test()