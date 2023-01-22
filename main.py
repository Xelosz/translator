import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Concatenate, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Loading the input data
with open("small_vocab_fr", "r") as f:
    source_sentences = f.readlines()

with open("small_vocab_en", "r") as f:
    target_sentences = f.readlines()

# Instantiating a tokenizer for the source and target
source_tokenizer = Tokenizer()
source_tokenizer.fit_on_texts(source_sentences)
source_vocab_size = len(source_tokenizer.word_index) + 1
source_sequences = source_tokenizer.texts_to_sequences(source_sentences)
source_padded = pad_sequences(source_sequences, padding='post')

target_tokenizer = Tokenizer()
target_tokenizer.fit_on_texts(target_sentences)
target_vocab_size = len(target_tokenizer.word_index) + 1
target_sequences = target_tokenizer.texts_to_sequences(target_sentences)
target_padded = pad_sequences(target_sequences, padding='post')

# Creating the model
encoder_inputs = tf.keras.layers.Input(shape=(None,))
encoder_embedding = Embedding(input_dim=source_vocab_size, output_dim=64)(encoder_inputs)
encoder_lstm_f = LSTM(64, return_sequences=True, go_backwards=False, name='encoder_lstm_f')(encoder_embedding)
encoder_lstm_b = LSTM(64, return_sequences=True, go_backwards=True, name='encoder_lstm_b')(encoder_embedding)
encoder_states = Concatenate(axis=-1)([encoder_lstm_f, encoder_lstm_b])
encoder_states = Lambda(lambda x: x[:, :64])(encoder_states)
encoder_states = [encoder_states, encoder_states]


decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = Embedding(input_dim=target_vocab_size, output_dim=64)(decoder_inputs)
decoder_lstm = LSTM(64, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Training the model
model.fit([source_padded, target_padded[:, :-1]], target_padded[:, 1:], epochs=100, batch_size=64)

# Using the model for translation
test_sentence = "I am going to the park."
test_sequence = source_tokenizer.texts_to_sequences([test_sentence])
test_padded = pad_sequences(test_sequence, padding='post')

encoder_model = Model(encoder_inputs, encoder_states)

decoder_states_inputs = [tf.keras.layers.Input(shape=(1, 2*64)), tf.keras.layers.Input(shape=(1, 2*64))]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

encoder_outputs = encoder_model.predict(test_padded)

eos = target_tokenizer.word_index['<eos>']

target_text = '<sos>'

while True:
    target_sequence = target_tokenizer.texts_to_sequences([target_text])
    target_padded = pad_sequences(target_sequence, padding='post')
    decoder_outputs, h, c = decoder_model.predict([target_padded] + encoder_outputs)
    output_token = tf.argmax(decoder_outputs[0, -1, :]).numpy()
    if output_token == eos:
        break
    target_text += ' ' + target_tokenizer.index_word[output_token]

print(target_text)

# Saving the model to an h5 file
model.save('translation_model.h5')


