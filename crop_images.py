from PIL import Image
import glob
import os
import tensorflow as tf

tf.app.flags.DEFINE_string('train_dir_unprocessed', 'data/train_unprocessed',
                           'Training data directory of unprocessed jpegs.')

tf.app.flags.DEFINE_string('train_dir_processed', 'data/train',
                           'Training data directory of processed jpegs.')

tf.app.flags.DEFINE_string('validation_dir_unprocessed', 'data/validation_unprocessed',
                           'Validation data directory of unprocessed jpegs.')

tf.app.flags.DEFINE_string('validation_dir_processed', 'data/validation',
                           'Validation directory of processed jpegs.')

tf.app.flags.DEFINE_integer('crop_size', 150,
                           'Size of cropped image.')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
tf.app.flags.DEFINE_string('labels_file', 'labels_file.txt', 'Labels file')

FLAGS = tf.app.flags.FLAGS

def crop_image(filename, save_filename):
	im = Image.open(filename)

	# Get current image dimensions.
	width, height = im.size

	# Set new image dimensions.
	new_width = FLAGS.crop_size
	new_height = FLAGS.crop_size

	# Get cropping coordinates.
	left = (width - new_width)/2
	top = (height - new_height)/2
	right = (width + new_width)/2
	bottom = (height + new_height)/2

	# Crop image.
	im_cropped = im.crop((left, top, right, bottom))

	# Save cropped image.
	im_cropped.save( save_filename, 'JPEG' )

def crop_images_in_directory(data_dir, save_dir, unique_labels):
	
	filenames = []

	# Construct the list of JPEG files.
	for text in unique_labels:
		jpeg_file_path = '%s/%s/*' % (data_dir, text)
		matching_files = glob.glob(jpeg_file_path)
		filenames.extend(matching_files)

	print("Found " + str(len(filenames)) + " files to crop.")

	# Crop and save JPEG files.
	for filename in filenames:
		save_filename = save_dir + filename[len(data_dir)::]
		crop_image(filename, save_filename)

def main():
	unique_labels = [l.strip() for l in tf.gfile.FastGFile(
      FLAGS.labels_file, 'r').readlines()]

	# Crop training images to be square.
	print("Cropping training images.")
	crop_images_in_directory(FLAGS.train_dir_unprocessed,
							FLAGS.train_dir_processed,
							unique_labels)
	print("Finished cropping training images.")

	# Crop validation images to be square.
	print("Cropping validation images.")
	crop_images_in_directory(FLAGS.validation_dir_unprocessed,
							FLAGS.validation_dir_processed,
							unique_labels)
	print("Finished cropping validation images.")


main()