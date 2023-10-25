# train_portal_ai.py
# author: Diego Magdaleno
# Train a GPT2 model (from gpt2.py) on the portal ai text lines and
# save it.
# Tensorflow 2.4
# Python 3.7
# Windows/MacOS/Linux


import os
import json
import string
import random
import gpt2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


def main():
	# Index all the files from the openwebtext corpus.
	print("Indexing training files...")
	characters = ["GLaDOS", "Wheatley"]
	character_path = "./" + characters[0]
	text_files = [character_path + "/" + file 
					for file in os.listdir(character_path)
					if file.endswith(".txt") 
					and valid_text(character_path + "/" + file)]
	batch_size = 128
	print("All training data indexed.")

	# Create a dataset from the text fields.
	print("Initializing dataset...")
	random.shuffle(text_files)
	text_ds = tf.data.TextLineDataset(text_files)
	text_ds = text_ds.shuffle(buffer_size=256)
	text_ds = text_ds.batch(batch_size)
	print("Dataset initialized.")

	# Set all text to lowercase and handle punctuation.
	# @param: input_string, the input string of the data.
	# @return: returns a cleaned version of the string.
	def custom_standardization(input_string):
		lowercased = tf.strings.lower(input_string)
		stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
		return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")

	def prepare_lm_inputs_labels(text):
		text = tf.expand_dims(text, -1)
		tokenized_sentences = vectorize_layer(text)
		x = tokenized_sentences[:, :-1]
		y = tokenized_sentences[:, 1:]
		return x, y

	# Intialize hyperparameters.
	vocab_size = 20000
	context_size = 80

	# Create a vectorization layer and adapt it to the text.
	print("Creating text vectorization and cleaning dataset...")
	vectorize_layer = TextVectorization(
		standardize=custom_standardization,
		max_tokens=vocab_size - 1,
		output_mode="int",
		output_sequence_length=context_size + 1
	)
	vectorize_layer.adapt(text_ds)
	vocab = vectorize_layer.get_vocabulary() # Get words back from token indices
	text_ds = text_ds.map(prepare_lm_inputs_labels)
	text_ds = text_ds.prefetch(tf.data.experimental.AUTOTUNE)
	print("Done.")

	# Load in pretrained GPT2 model.
	print("Loading pretrained GPT2 model...")
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	pretrained_model_path = ".//"
	loaded_model = gpt2.GPT2()
	loaded_model.load(pretrained_model_path)
	loaded_model.loss = [loss_fn, None]
	context_size = loaded_model.context_size
	print("Done.")

	# Load in the vocabulary mappings from the pretrained model.
	print("Loading pretrained vocabulary...")
	with open(pretrained_model_path + "vocab.json", "r") as v_file:
		pretrained_vocab = json.load(v_file)
	print("Vocabulary loaded.")

	# Training hyperparameters.
	# Verbosity. 0 => silent, 1 => progress bar, 2 => 1 line per epoch.
	#verbose = 2
	verbose = 1
	epochs = 25

	# Tokenize starting prompt.
	print("Tokenizing starting prompt...")
	word_to_index = {}
	'''
	for index, word in enumerate(vocab):
		word_to_index[word] = index
	'''
	# Load pretrained vocabulary to the vocabulary for this model.
	# Merge any new vocabulary from the training dataset together.
	word_to_index = pretrained_vocab
	#max_val = max(list({v: k for k, v in pretrained_vocab.items()}.keys())) + 1
	max_val = len(pretrained_vocab) + 1
	for word in vocab:
		if word not in word_to_index:
			word_to_index[word] = max_val
			max_val += 1

	start_prompt = "This next test"
	start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
	num_tokens_generated = 256
	text_gen_callback = gpt2.TextGenerator(num_tokens_generated, start_tokens, 
											vocab, context_size)
	print("Created text generator callback.")

	# Train GPT2 on data.
	print("Starting training...")
	model_name = characters[0]
	if not os.path.exists("./" + model_name + "_model_checkpoints"):
		os.mkdir("./" + model_name + "_model_checkpoints")
	model_checkpoint = keras.callbacks.ModelCheckpoint(
		"./" + model_name + "_model_checkpoints", save_weights_only=True,
	)
	new_gpt.gpt_model.fit(
		text_ds, verbose=verbose, epochs=epochs, batch_size=64,
		callbacks=[text_gen_callback, model_checkpoint]
	)
	print("Model trained.")

	# Save GPT2 model.
	save_path = "./gpt2_xxs"
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	new_gpt.save(save_path)

	# Load GPT2 model.
	load_gpt = gpt2.GPT2()
	load_gpt.load(save_path)
	print(load_gpt.gpt_model.summary())

	# Exit the program.
	exit(0)


# Remove files that contain characters not within unicode.
def valid_text(file):
	# Open the file and read its contents,
	with open(file, "r") as f:
		file_contents = f.read()
	
	# Go through the string contents of the file and if a character is
	# outside the range of readable UTF-8 values, return False (file is
	# not valid).
	for char in file_contents:
		if ord(char) > 126:
			return False
	
	# Return True (file is valid).
	return True


if __name__ == '__main__':
	main()