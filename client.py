from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences




# Load the model from the h5 file
loaded_model = load_model('translation_model.h5')

# Define the test sentence
test_sentence = "I am going the store."
#setting
source_tokenizer = Tokenizer()
source_tokenizer.fit_on_texts(test_sentence)
source_sequences = source_tokenizer.texts_to_sequences(test_sentence)
max_length_target = max([len(seq) for seq in test_sentence])
# Tokenize the test sentence
test_sequence = source_tokenizer.texts_to_sequences([test_sentence])

# Pad the test sequence to match the max length of the target sequences
test_padded = pad_sequences(test_sequence, maxlen=max_length_target)

# Make the prediction using the loaded model
predictions = loaded_model.predict(test_padded)
target_tokenizer = Tokenizer()
target_tokenizer.fit_on_texts(test_sentence)
# Convert the prediction back to the original text
predictions = target_tokenizer.sequences_to_texts(predictions)

print(predictions)
