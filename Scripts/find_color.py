from PIL import Image
import numpy as np

img = Image.open('../map_data/provinces.png')

img_array = np.array(img)

color = (255, 22, 253)
coords = np.where(np.all(img_array == color, axis=-1))

found_coords = list(zip(coords[1], coords[0]))

print(found_coords)