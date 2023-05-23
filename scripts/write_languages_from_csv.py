import pandas as pd

def write_column_loc(df, column):
	ids = df["Province ID"]
	locs = df[column]
	suffix = column if column != "English" else ""
	with open(f"{column}_province_names_l_english.yml", "w") as f:
		f.write(f"# Generated {column} province name localization")
	with open(f"{column}_province_names_l_english.yml", "a") as f:
		for i, pid in enumerate(ids):
			f.write(f"PROV{pid}{suffix}:0 \"{locs[i]}\"\n")

def main(csvfile):
	data = pd.read_csv(csvfile)
	df = pd.DataFrame(data)

	write_column_loc(df, "English")
	write_column_loc(df, "Eageli")
	write_column_loc(df, "Ytali")
	write_column_loc(df, "Weagli")

if __name__ == '__main__':
	main("omniluxia province names.csv")
