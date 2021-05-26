# prep_train_tts.py
# author: Diego Magaleno
# General script that prepares all audio and text files to train on
# tacotron 2. The model is then trained and saved appropriately.
# Python 3.7
# Tensorflow 2.4
# Windows/MacOS/Linux


import os
import subprocess
from shutil import copyfile


def main():
	# Start by check to see if the tacotron 2 github repo has been downloaded.
	# Download it if it has not.
	print("Checking for Tacotron2...")
	if "tacotron2" not in os.listdir():
		print("Could not find Tacotron2. Downloading from github.")
		download = subprocess.Popen("git clone https://github.com/NVIDIA/tacotron2", 
			shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
		)
		output, error = download.communicate()

		# If there are any unexpected error messages, print them and exit.
		if error.decode("utf-8") != "Cloning into 'tacotron2'...\n":
			print(error.decode("utf-8"))
			print("Failed to download Tacotron2. Exiting program.")
			exit(1)

		# Output expected error and output messages.
		if output.decode("utf-8") != "":
			print(output)
		print(error.decode("utf-8"))
		print("Tacotron2 has been successfully downloaded.")
	print("Tacotron2 found.")

	characters = ["GLaDOS", "Wheatley"]
	character = characters[0]

	# Go through the character audio and text for that character.
	character_path = "./" + character + "/"
	audio_files = [character_path + wav for wav in os.listdir(character_path)
					if wav.endswith(".wav")]
	text_files = [character_path + txt for txt in os.listdir(character_path)
					if txt.endswith(".txt")]

	# Create a folder to store the wav files for the character in the tacotron
	# repository if it doesn't already exist. Remove any files in there from
	# the last one.
	taco_wavs_folder = "./tacotron2/wavs/"
	if not os.path.exists(taco_wavs_folder):
		os.mkdir(taco_wavs_folder)
	if len(os.listdir(taco_wavs_folder)) != 0:
		contents = os.listdir(taco_wavs_folder)
		for obj in contents:
			os.remove(taco_wavs_folder + obj)

	# Copy over character wav files to the wav folder in tacotron
	# repository.
	#for audio_file in audio_files:
	#	copyfile(audio_file, taco_wavs_folder + "/" + audio_file.split("/")[-1])
	for audio_file_idx in range(len(audio_files)):
		copyfile(audio_files[audio_file_idx], 
			taco_wavs_folder + str(audio_file_idx) + ".wav"
		)

	# Format the corresponding transcript text files accordingly. Add
	# that text to a string to write the filelist txt file.
	filelist_save = "./tacotron2/filelists/" + character + "_filelist.txt"
	filelist_text = ""
	for text_file_idx in range(len(text_files)):
		formatted_text = clean_text(text_files[text_file_idx])
		header = "wavs/" + str(text_file_idx) + ".wav|"
		filelist_text += header + formatted_text + "\n"
	with open(filelist_save, "w+") as filelist:
		filelist.write(filelist_text)

	# Run training script train.py in tacotron2 repository to train
	# model.
	output_dir = character + "_saved_checkpoints"
	log_dir = character + "_logs"
	n_gpus = 0
	n
	train = subprocess.Popen("python ./tacotron2/train.py -o")

	# Exit the program.
	exit(0)


def clean_text(file):
	with open(file, "r", encoding="cp1252") as read_file:
		file_text = read_file.read()
	file_text = file_text.replace("\n", " ").encode("utf-8").decode("utf-8")
	if not file_text.endswith("."):
		file_text += "."
	return file_text


if __name__ == '__main__':
	main()
