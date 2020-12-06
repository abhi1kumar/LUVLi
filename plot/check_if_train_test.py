

import os

filename = "plot/AFLW_ours_images_for_plotting_2.txt"
filename2 = "plot/AFLW_ours_images_for_plotting_2.txt"

# Read a file list in python
with open(filename) as f:
    image_paths = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
image_paths = [x.strip() for x in image_paths]

num_files = len(image_paths)
output_path = []

for i in range(num_files):
	rel_path      = image_paths[i]
	
	# Do some replacements
	rel_path = rel_path.replace("Frontal", "frontal")
	rel_path = rel_path.replace("LeftHalf", "lefthalf")
	rel_path = rel_path.replace("RightHalf", "righthalf")
	rel_path = rel_path.replace("Right", "right")
	rel_path = rel_path.replace("Left", "left")

	img_full_path = os.path.join("bigdata1/zt53/data/face/aflw_ours_organized", rel_path)

	# check if file exists, if not then change train to test
	if os.path.exists(img_full_path):
		output_path.append(rel_path)
	else:
		# file not found
		if "trainset" in rel_path:
			rel_path = rel_path.replace("trainset","testset")
		
		if "testset" in rel_path:
			rel_path = rel_path.replace("testset","trainset")
		
		img_full_path = os.path.join("bigdata1/zt53/data/face/aflw_ours_organized", rel_path)
		# Again check if file is there or not
		if os.path.exists(img_full_path):
			output_path.append(rel_path)
		else:
			print("{} not found".format(rel_path))

print(output_path)

# Now write list to the file
with open(filename2, 'w') as f:
    for item in output_path:
        f.write("%s\n" % item)
