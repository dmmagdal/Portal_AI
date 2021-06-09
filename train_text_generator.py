# train_text_generator.py
# author: Diego Magdaleno
# Train a text generation model (pick from transformers from
# huggingface) for one of the given characters. This program will use
# the HappyTransformer module to wrap around the huggingface module.
# Tensorflow 2.4
# Python 3.7
# Windows/MacOS/Linux


import gc
import os
import happytransformer as ht
'''
#from transformers import ByteLevelBPETokenizer
from transformers import TFXLNetModel
from transformers import GPT2Tokenizer, TFGPT2Model
#from transformers import GPTNeoForCausalLM
'''


def main():
	# Load the characters.
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

	# Prmpts to test a few text generations.
	prompt1 = "Oh, it's you. It's been a long time"
	prompt2 = "You know, here in Aperture Science"

	# Huggingface GPT-2 variation (distilgpt2, gpt2-medium, gpt2-large,
	# gpt2-xl, openai-gpt, EleutherAI/gpt-neo-125M, xlnet-base-cased,
	# xlnet-large-cased).
	# Could not download the following models:
	# 1) EleutherAI/gpt-neo-2.7B
	# Could not train the following models:
	# 1) gpt2-medium (Lenovo Laptop)
	# 2) gpt2-large (Lenovo Laptop)
	# 3) gpt2-xl (Lenovo Laptop, Dell Desktop)
	# 4) EleutherAI/gpt-neo-1.3B (Lenovo Laptop)
	# 5) EleutherAI/gpt-neo-2.7B (Lenovo Laptop, Dell Desktop)
	# 6) xlnet-based-cased (Lenovo Laptop, Dell Desktop)
	# 7) xlnet-large-cased (Lenovo Laptop, Dell Desktop)
	# Note that for training the GPT-Neo 1.3B model on the Dell
	# Desktop, it is best to run the training when it is the only
	# active program on the machine.
	#models = {"gpt-2": ["openai-gpt", "gpt-2", "distilgpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], 
	#			"gpt-neo": ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"],
	#			"xlnet": ["xlnet-base-cased", "xlnet-large-cased"]}
	models = {"gpt-2": ["openai-gpt", "gpt-2", "distilgpt2", "gpt2-medium", "gpt2-large"], 
				"gpt-neo": ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B"],
				"xlnet": ["xlnet-base-cased", "xlnet-large-cased"]}
	

	# Test training models.
	for m in models:
		for n in models[m]:
			# Instiate model and train it.
			model = ht.HappyGeneration(model_type=m, model_name=n)
			print("Training {}/{}:".format(m, n))
			model.train(character + "_compiled_lines.txt")

			# Print the model's responses to the given prompts.
			print("{}/{} Text Responses:".format(m, n))
			response1 = model.generate_text(prompt1, args=gen_settings).text
			print("Prompt: " + prompt1)
			print("Output: " + response1)

			response2 = model.generate_text(prompt2, args=gen_settings).text
			print("Prompt: " + prompt2)
			print("Output: " + response2)

			# Save the model.
			model_save = "_AI_" + n.replace("/", "_")
			model.save(model_save)

			# Do some garbage collection on memory.
			gc.collect()

	# Huggingface GPT-2 Text generator.
	text_gen = ht.HappyGeneration()
	text_gen.train(character + "_compiled_lines.txt")

	'''
	# Huggingface GPT-Neo text generators.
	neo_gen1_3b = ht.HappyGeneration(model_type="GPT-NEO", model_name="EleutherAI/gpt-neo-1.3B")
	neo_gen1_3b.train(character + "_compiled_lines.txt")
	neo_gen2_7b = ht.HappyGeneration(model_type="GPT-NEO", model_name="EleutherAI/gpt-neo-2.7B")
	neo_gen2_7b.train(character + "_compiled_lines.txt")
	'''

	'''
	# Text generations with GPT-2.
	response1 = text_gen.generate_text(prompt1, args=gen_settings).text
	print("GPT-2 Text Responses:")
	print("Prompt: " + prompt1)
	print("Output: " + response1)

	response2 = text_gen.generate_text(prompt2, args=gen_settings).text
	print("Prompt: " + prompt2)
	print("Output: " + response2)
	'''

	'''
	# Text generations with GPT-Neo 1.3B.
	response1_neo_13 = neo_gen1_3b.generate_text(prompt1, args=gen_settings).text
	print("GPT-2 Text Responses:")
	print("Prompt: " + prompt1)
	print("Output: " + response1_neo_13)

	response2_neo_13 = neo_gen1_3b.generate_text(prompt2, args=gen_settings).text
	print("Prompt: " + prompt2)
	print("Output: " + response2_neo_13)

	# Text generations with GPT-Neo 2.7B.
	response1_neo_27 = neo_gen2_7b.generate_text(prompt1, args=gen_settings).text
	print("GPT-2 Text Responses:")
	print("Prompt: " + prompt1)
	print("Output: " + response1_neo_27)

	response2_neo_27 = neo_gen2_7b.generate_text(prompt2, args=gen_settings).text
	print("Prompt: " + prompt2)
	print("Output: " + response2_neo_27)
	'''

	# Save the model.
	model_save_gpt2 = character + "_AI_GPT-2"
	text_gen.save(model_save_gpt2)
	'''
	model_save_gpt_neo_13 = character + "_AI_GPT-Neo_1B"
	neo_gen1_3b.save(model_save_gpt_neo_13)
	model_save_gpt_neo_27 = character + "_AI_GPT-Neo_2B"
	neo_gen2_7b.save(model_save_gpt_neo_27)
	'''

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