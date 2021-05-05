# ai_downloader.py
# author: Diego Magdaleno
# Download voice lines (wav audio files and associated transcription)
# from the Portal wiki for AIs GladOS and Wheatley.
# Python 3.7
# Windows/MacOS/Linux


import os
import requests
import multiprocessing as mp
from bs4 import BeautifulSoup as bsoup
from tqdm import tqdm


def main():
	# Initialize the list of characters whose voice lines this program
	# is able to download.
	characters = ["GLaDOS", "Wheatley"]

	# Here are some other valid character voice lines that can be
	# downloaded from the Portal wiki. A few things to note:
	# 1) Ap-Sap -> additional Wheatley voices for the Ap-Sap device in
	#	Team Fortress 2.
	# 2) Core -> consists of all the personality cores. Additional
	#	preprocessing will be required to sort through the different
	#	personalities.
	# 3) Oracle_Turret & Defective_Turret -> similar to Ap-Sap having
	#	additional Wheatley lines, Oracle_Turret and Defective_Turret
	#	can be considered extra voice lines for the Turret in general
	#	but Defective_Turret's voice is different from the other two.
	#	Additional preprocessing would be advised if any of these
	#	characters would have their data combined for one entity.
	other_valid_characters = ["Ap-Sap", "Turret", "Core", "Oracle_Turret",
								"Defective_Turret", "Caroline", "Announcer",
								"Cave_Johnson"]

	# Iterate through that list of characters and download the voice
	# lines and transcripts.
	#for char in characters:
	#	download_files(char)

	# Optional multiprocessing.
	pool = mp.Pool(processes=len(characters))
	pool.map(download_files, tuple(characters))

	# Exit the program.
	exit(0)


# Goes to the specific voice lines page for a character in the Portal
# series and downloads the respective audio and text.
# @param: character, (str) the character whose lines are being 
#	downloaded.
# @return: returns nothing.
def download_files(character):
	# Create a folder to store the data if it does not exist already.
	folder_path = "./" + character + "/"
	if not os.path.exists(folder_path):
		os.mkdir(folder_path)

	# Send a request to the site and pass through beautifulsoup.
	url = "https://theportalwiki.com/wiki/" + character + "_voice_lines"
	
	response = requests.get(url)
	if response.status_code != 200:
		print("Could not get retrieve page for " + character +\
			" Bad request status_code.", flush=True)
		print("status_code: " + str(response.status_code), flush=True)
		return

	page_soup = bsoup(response.text, features="lxml")

	# Isolate the proper HTML tags that contain the voice lines and
	# text.
	line_items = [item for item in page_soup.find_all("li") 
					if "class" not in item.attrs and "id" not in item.attrs
					and item.find("a") and item.find("a")["href"].endswith(".wav")]

	# Iterate through the tags.
	print("Downloading " + character + " voice lines...")
	for i in tqdm(range(len(line_items))):
		# Isolate the url for the audio file.
		audio_url = line_items[i].find("a")["href"]
		audio_filename = audio_url.split("/")[-1]
		
		# Send a request for that audio file and save it.
		audio_response = requests.get(audio_url)
		with open(folder_path + audio_filename, "wb+") as audio:
			audio.write(audio_response.content)

		# Save the text lines to a file.
		#print(line_items[i].text.strip("\n")[1:-1])
		processed_text = line_items[i].text.strip("\n")[1:-1]
		if processed_text.count("\"") == 2:
			processed_text = processed_text.split("\"")[1]
		text_filename = audio_filename[:-4] + ".txt"
		with open(folder_path + text_filename, "w+") as txt:
			txt.write(processed_text)

	# Manually clean the remaining texts. They will be listed here.
	text_files = [folder_path + text for text in os.listdir(folder_path) 
				if text.endswith(".txt")]
	for text in text_files:
		with open(text, "r", encoding="cp1252") as f:
			file_lines = f.read()
		if file_lines.count("\"") != 0:
			print(text)
			print(file_lines)
			print("-"*72)

	# Return the function.
	return
	

if __name__ == '__main__':
	main()