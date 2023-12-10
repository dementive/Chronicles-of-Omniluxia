from PIL import Image
import numpy as np

img = Image.open('../map_data/provinces.png')

img_array = np.array(img)

color = (131, 12, 184)
coords = np.where(np.all(img_array == color, axis=-1))

found_coords = list(zip(coords[0], coords[1]))

print(found_coords)