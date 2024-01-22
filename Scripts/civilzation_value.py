import re
import os


"""
This script is used to update the civilization value for all provinces in setup to the civilization value cap for optimal balance at game start.

Steps to use:

1. You'll need to open up imperator in debug mode and run an event with this effect in it

every_province = {
	limit = {
		is_sea = no
	}
	debug_log = "~[THIS.GetProvince.GetId];[THIS.GetProvince.GetLocalCivilizationCap]!"
}

2. Go to your imperator logs folder and move the debug.log file into the same folder as this script.

3. Update the PROVINCE_SETUP_FOLDER variable with the path to your mods "setup\\provinces" folder and then run the script

"""

# Path to the directory province setup files
PROVINCE_SETUP_FOLDER = "C:\\...\\setup\\provinces"

if __name__ == '__main__':
	with open("debug.log", "r") as file:
		lines = file.readlines()

	lines_string = str()
	for line in lines:
		lines_string += line
	matches = re.findall(r"~(\d+);(\d+)\.0!", lines_string)

	prov_id_to_civ_value_dict = dict()
	
	for i in matches:
		prov_id_to_civ_value_dict[i[0]] = i[1]

	for filename in os.listdir(PROVINCE_SETUP_FOLDER):
		if filename.endswith('.txt'):
			file_path = os.path.join(PROVINCE_SETUP_FOLDER, filename)
			
			with open(file_path, 'r') as file:
				file_contents = file.read()
			
			blocks = re.findall(r'(\d+)=(\{.*?\n\})', file_contents, re.DOTALL)

			# Update the civilization_value for each block
			for block_id, block in blocks:
				if block_id in prov_id_to_civ_value_dict:
					updated_block = re.sub(r'(civilization_value=)(\d+)', r'\g<1>{}'.format(prov_id_to_civ_value_dict[block_id]), block)
					file_contents = file_contents.replace(block, updated_block)
			
			with open(file_path, 'w') as file:
				file.write(file_contents)
