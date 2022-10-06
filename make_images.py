import time
from PIL import Image
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt
import sys 

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")

with open('prompts.txt') as f:
	for prompt in f.read().split('\n'):
		print(prompt)
		images = model.text_to_image(prompt, batch_size=4)

		for i, img in enumerate(images):
		  im = Image.fromarray(img)
		  im.save(f"{prompt[:240]}{i}.png")
