import os
import pandas as pd
import numpy as np
from PIL import Image
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--which_back', '-which_back',
		default='water',
		choices=['water', 'land'],
		help="Which background",
		type=str)
	args = vars(parser.parse_args())
	which_back = args['which_back']
	print(which_back)
	# --- Get the list of experiment images
	imagepath = f'/nfs/turbo/coe-rbg/waterbirds/{which_back}_easy/'
	list_of_experiment_images = [
			f'{imagepath}{f}' for f in os.listdir(imagepath)
			if os.path.isfile(os.path.join(imagepath, f))
	]
	list_of_experiment_images = sorted(list_of_experiment_images)

	# -- create a dataframe of experiment images
	image_df = pd.DataFrame({'experiment_image': list_of_experiment_images})
	image_df['original_image'] = ''

	# --- Get list of original images
	if which_back == 'water':
		imagepath = '/nfs/turbo/coe-rbg/data_large/b/beach/'
		list_of_original_images = [
				f'{imagepath}{f}' for f in os.listdir(imagepath)
				if os.path.isfile(os.path.join(imagepath, f))
		]
	else:
		imagepath = '/nfs/turbo/coe-rbg/data_large/f/forest/broadleaf/'
		list_of_original_images = [
				f'{imagepath}{f}' for f in os.listdir(imagepath)
				if os.path.isfile(os.path.join(imagepath, f))
		]

		imagepath = '/nfs/turbo/coe-rbg/data_large/b/bamboo_forest/'
		list_of_original_images2 = [
				f'{imagepath}{f}' for f in os.listdir(imagepath)
				if os.path.isfile(os.path.join(imagepath, f))
		]

		list_of_original_images = list_of_original_images + list_of_original_images2
		print(len(list_of_original_images))

	list_of_original_images = sorted(list_of_original_images)


	# --- commence the matching
	for eid, experiment_image in enumerate(list_of_experiment_images):
		print(f'{eid} /{image_df.shape[0]}')
		exp_img_data = Image.open(experiment_image)
		exp_img_data = np.asarray(exp_img_data)
		for original_image in list_of_original_images:
				orig_img_data = Image.open(original_image)
				orig_img_data = np.asarray(orig_img_data)
				if np.array_equal(exp_img_data, orig_img_data):
						image_df[(image_df.experiment_image == experiment_image)]['original_image'] = orignal_image
						list_of_original_images.remove(original_image)
						print(image_df[(image_df.original_image != '')])
						break
		if eid % 100 == 0:
				image_df.to_csv(f'/nfs/turbo/coe-rbg/mmakar/single_shortcut/{which_back}back_image_match.csv', index=False)


