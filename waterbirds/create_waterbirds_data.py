# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Creates waterbirds datasets."""
import os
import glob
import shutil

from absl import app

# TODO dont hard code directories
OUTPUT_DIR = '/data/ddmg/slabs/waterbirds/places_data'
BIRDS_DIR = '/data/ddmg/slabs/CUB_200_2011'
PLACES_DIR = '/data/ddmg/slabs/data_large'

WATER_PLACES = ['o/ocean', 'l/lake/natural']
BEACH_PLACES = ['b/beach']
LOCATION_TO_SPECS = {
	# 'water': ['o/ocean', 'l/lake/natural'],
	# 'land': ['b/bamboo_forest', 'f/forest/broadleaf'],
	'beach': ['b/beach']
}


def copy_place_images(location, subdirs):
	place_id = 0
	for place in subdirs:
		place_files = glob.glob(f'{PLACES_DIR}/{place}/*.jpg')
		for place_file in place_files:
			shutil.copy(place_file, f'{OUTPUT_DIR}/{location}/image_{place_id}.jpg')
			place_id = place_id + 1


def main(args):
	del args

	if not os.path.exists(f'{OUTPUT_DIR}'):
		os.mkdir(f'{OUTPUT_DIR}')

	for location, subdirs in LOCATION_TO_SPECS.items():
		if not os.path.exists(f'{OUTPUT_DIR}/{location}'):
			os.mkdir(f'{OUTPUT_DIR}/{location}')

		print(location, subdirs)
		copy_place_images(location, subdirs)


if __name__ == '__main__':
	app.run(main)
