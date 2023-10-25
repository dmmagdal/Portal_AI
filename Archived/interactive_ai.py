# interactive_ai.py
# author: Diego Magdaleno
# An interactive program that allows a user to enter a prompt for a
# portal ai character and will 
# Python 3.7
# Windows/MacOS/Linux


import os
import happytransformer as ht


def main():
	# Load the model.
	characters = ["GLaDOS", "Wheatley"]
	character = characters[0]
	model_save = character + "_AI"
	text_gen = ht.HappyGeneration(load_path=model_save)

	# Initialize text generation settings.
	gen_settings = ht.GENSettings()
	gen_settings.max_length = get_maxlen(character)
	gen_settings.early_stopping = True
	gen_settings.no_repeat_ngram_size = 2

	# Infinite loop. Keep prompting the AI until entering "EXIT".
	prompt_text = ""
	#while prompt_text != "EXIT"
	while True:
		# Get the prompt text from the user.
		prompt_text = input("PROMPT  >> ")

		# Exit the loop if the input prompt text is "EXIT".
		if prompt_text == "EXIT":
			break

		# Feed the prompt to the model and output it.
		output_text = text_gen.generate_text(
			prompt_text, args=gen_settings
		).text
		print("OUTPUT >> " + output_text)
		print()

	# Exit the program.
	exit(0)


# Get the longest length of text that a character says.
def get_maxlen(character):
	# List out all text files for the character.
	text_files = [character + "/" + file 
					for file in os.listdir(character)
					if file.endswith(".txt")
					and valid_text(character + "/" + file)]

	# Compile all texts into a single file.
	compiled_text = ""
	for file in text_files:
		with open(file, "r", encoding="cp1252") as f:
			compiled_text += "\n" + f.read()

	# Go through the compiled text and find the length of the longest
	# line in that character's text.
	max_length = 0
	for line in compiled_text.split("\n"):
		max_length = max(len(line.split()), max_length)

	# Return the max length of a 
	return max_length


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