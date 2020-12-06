import numpy as np

a  = np.load('abhinav_model_dir/run_441_test/means.npy')
gt = np.load('abhinav_model_dir/run_441_test/ground_truth.npy')
t  = a-gt

max_box = np.max(gt, 1)
min_box = np.min(gt, 1)
width_height_gd = max_box - min_box
print(width_height_gd.shape)
scale = np.sqrt(width_height_gd[:,0] *width_height_gd[:,1])
noise = 0.0

bbox = np.sqrt((width_height_gd[:,0] + np.random.uniform(0, noise * scale)) * (width_height_gd[:,1] + np.random.uniform(0, noise * scale)))
print(bbox.shape)

bbox = np.tile(np.expand_dims(bbox, axis=0).transpose(), 68)
print(bbox.shape)

error = np.linalg.norm(t.reshape(-1, 2), ord=2, axis=1).reshape(-1, 68)
temp  = error/bbox
print(np.mean(temp))

