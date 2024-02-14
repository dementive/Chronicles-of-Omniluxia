import re
import os

# Check province setup for cities that produce food


PROVINCE_SETUP_FOLDER = "..\\setup\\provinces"
FOOD_GOODS = [
	"grain",
	"fish",
	"cattle",
	"vegetables",
	"fruits",
	"honey",
	"dates",
]

if __name__ == '__main__':
	for filename in os.listdir(PROVINCE_SETUP_FOLDER):
		if filename.endswith('.txt'):
			file_path = os.path.join(PROVINCE_SETUP_FOLDER, filename)
			
			with open(file_path, 'r') as file:
				file_contents = file.read()
			
			blocks = re.findall(r'(\d+)=(\{.*?\n\})', file_contents, re.DOTALL)

			# Update the civilization_value for each block
			for block_id, block in blocks:
				if 'city' in block or 'city_metropolis' in block:
					match = re.search(r'trade_goods="(.*?)"', block)
					if not match:
						break

					trade_good = match.group(1)
					if trade_good in FOOD_GOODS:
						print(f"{block_id} - {match.group(1)}")
			
			with open(file_path, 'w') as file:
				file.write(file_contents)
