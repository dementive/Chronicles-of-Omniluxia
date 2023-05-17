import re
from pathlib import Path

# Check for missing fields in province setup

class ProvinceChecker:
    def __init__(self, text):
        self.data = {}
        self.key = ""
        for i, line in enumerate(text.split("\n")):
            match = re.match(r"(.+?)\s?=\s?(.+)", line)
            if match:
                key = match.group(1).strip()
                if i == 0:
                    self.key = key
                value = match.group(2).strip()
                self.data[key] = value
        setattr(self, "data", self.data)

    def check_province(self):
        terrain = True
        culture = True
        religion = True
        trade_goods = True
        civilization_value = True
        barbarian_power = True
        province_rank = True

        try:
            self.data["terrain"]
        except KeyError:
            terrain = False
        try:
            self.data["culture"]
        except KeyError:
            culture = False
        try:
            self.data["religion"]
        except KeyError:
            religion = False
        try:
            self.data["trade_goods"]
        except KeyError:
            trade_goods = False
        try:
            self.data["civilization_value"]
        except KeyError:
            civilization_value = False
        try:
            self.data["barbarian_power"]
        except KeyError:
            barbarian_power = False
        try:
            self.data["province_rank"]
        except KeyError:
            province_rank = False

        return (terrain, culture, religion, trade_goods, civilization_value, barbarian_power, province_rank)

if __name__ == '__main__':

    directory = "C:\\Users\\demen\\Documents\\Paradox Interactive\\Imperator\\mod\\Chronicles-of-Omniluxia\\setup\\provinces"

    for filename in Path(directory).iterdir():
        with open(filename, 'r', encoding="utf-8-sig") as file:
            text = file.read()

        provinces = re.findall(r'(\d+)=\{(.+?)\n\}', text, re.DOTALL)
        error_list = list()

        for i in provinces:
            broken = False
            province = ProvinceChecker(i[1])
            checked = province.check_province()

            if not checked[0]:
                broken = True
            if not checked[1]:
                broken = True
            if not checked[2]:
                broken = True
            if not checked[3]:
                broken = True
            if not checked[4]:
                broken = True
            if not checked[5]:
                broken = True
            if not checked[6]:
                broken = True

            # Check impassable_terrain entries
            try:
                if province.data["terrain"] == '"impassable_terrain"':
                    if not province.data["trade_goods"] == '""' or not province.data["province_rank"] == '""':
                        broken = True
            except KeyError:
                province_rank = False

            # Check ocean entires
            try:
                if province.data["terrain"] == '"ocean"' or province.data["terrain"] == '"coastal_terrain"' or province.data["terrain"] == '"riverine_terrain"':
                    if not province.data["trade_goods"] == '""' or not province.data["province_rank"] == '""':
                        broken = True
                    if not province.data["culture"] == '""' or not province.data["religion"] == '""':
                        broken = True
            except KeyError:
                province_rank = False

            if broken:
                error_list.append(i[0])

        if error_list:
            print(f"{filename.name} - {error_list}")
