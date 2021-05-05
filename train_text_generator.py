# train_text_generator.py
# author: Diego Magdaleno
# Train a text generation model (pick from transformers from
# huggingface) for one of the given characters. This program will use
# the HappyTransformer module to wrap around the huggingface module.
# Tensorflow 2.4
# Python 3.7
# Windows/MacOS/Linux


import os
import happytransformer as ht
'''
#from transformers import ByteLevelBPETokenizer
from transformers import TFXLNetModel
from transformers import GPT2Tokenizer, TFGPT2Model
#from transformers import GPTNeoForCausalLM
'''


def main():
	characters = ["GLaDOS", "Wheatley"]
	character = characters[0]

	# List out all text files to train byte-level Byte-pair encoding
	# tokenizer (same kind of tokenizer as GPT-2).
	text_files = [character + "/" + file 
					for file in os.listdir(character)
					if file.endswith(".txt")
					and valid_text(character + "/" + file)]

	# Compile all texts into a single file.
	compiled_text = ""
	for file in text_files:
		with open(file, "r", encoding="cp1252") as f:
			compiled_text += "\n" + f.read()
	with open(character + "_compiled_lines.txt", "w+") as chara_file:
		chara_file.write(compiled_text)

	# Set the maximum length of the GENSetting to be the same as the
	# max length of the longest line in the character's lines.
	max_length = 0
	for line in compiled_text.split("\n"):
		max_length = max(len(line.split()), max_length)

	# Initialize a text generator and train it.
	gen_settings = ht.GENSettings()
	gen_settings.max_length = max_length
	gen_settings.early_stopping = True
	gen_settings.no_repeat_ngram_size = 2
	text_gen = ht.HappyGeneration()
	text_gen.train(character + "_compiled_lines.txt")

	# Test a few text generations.
	prompt1 = "Oh, it's you. It's been a long time"
	response1 = text_gen.generate_text(prompt1, args=gen_settings).text
	print("Prompt: " + prompt1)
	print("Output: " + response1)

	prompt2 = "You know, here in Aperture Science"
	response2 = text_gen.generate_text(prompt2, args=gen_settings).text
	print("Prompt: " + prompt2)
	print("Output: " + response2)

	# Save the model.
	model_save = character + "_AI"
	text_gen.save(model_save)

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