import torch
from torch.utils import data
from os.path import join as pjoin
import os
from PIL import Image
import numpy as np
import json
import sys
from utils import *
import torchvision
import cv2
from .sequence_blacklist import sequence_blacklist

def read_rgb(data_split_dir, scene, id):
    path = os.path.join(data_split_dir, scene, "color", str(id) + ".jpg")
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def read_intrinsics(data_split_dir, scene, type="intrinsic_color"):
    cam_intr = np.loadtxt(os.path.join(data_split_dir, scene, "intrinsic", "{}.txt".format(type)), delimiter=' ')
    return cam_intr

def read_pose(data_split_dir, scene, id, verbose=True):
	cam_pose = np.loadtxt(os.path.join(data_split_dir, scene, "pose", str(id) + ".txt"), delimiter=' ')
	if not np.all(np.isfinite(cam_pose)):
		if verbose:
			print("Error: Non-finite pose for {}, id {}".format(scene, id))
		return None
	return cam_pose

# pos/neg crop_x crops on the left/right of the image
# pos/neg crop_y crops on the top/bottom of the image
def crop_image(img, crop_x, crop_y):
	h, w = img.shape[:2]
	if crop_x < 0:
		img = img[:, : w + crop_x]
	else:
		img = img[:, crop_x:]
	if crop_y < 0:
		img = img[: h + crop_y]
	else:
		img = img[crop_y:]
	return img

# pos/neg crop_x cropped on the left/right of the image
# pos/neg crop_y cropped on the top/bottom of the image
# neg crop_* (i.e., cropping on right/bottom) does not affect K
def crop_intrinsics(K, crop_x, crop_y):
    if crop_x > 0:
        K[0, 2] -= crop_x
    if crop_y > 0:
        K[1, 2] -= crop_y
    return K

def resize_intrinsics(K, fact_x, fact_y):
    K[0, 0] *= fact_x
    K[1, 1] *= fact_y
    K[0, 2] *= fact_x
    K[1, 2] *= fact_y
    return K

# convert from scannet (==colmap/opencv) to blender
def convert_pose_scannet_to_blender(pose):
	# transform from camera with view dir in +z to camera with view dir in -z
	cam2cam = np.eye(4)
	cam2cam[1, 1] *= -1
	cam2cam[2, 2] *= -1
	pose = pose @ cam2cam
	return pose

# adapted rel_camera_ray_encoding from camera_tools.py to take different focal lengths in x and y
def rel_camera_ray_encoding2(tform, im_size, focal_x, focal_y):
	# create camera ray encoding
	# assumes square images with same focal length on all axes, assume principle = 0.5
	cam_center = tform[:3,-1]
	cam_rot = tform[:3,:3]

	# find max limit from focal length
	max_pos_x = 1/(2*focal_x) # our images are normalizedto -1,1
	max_pos_y = 1/(2*focal_y) # our images are normalizedto -1,1

	# find rays for all pixels
	pix_size = 2/im_size
	x,y = np.meshgrid(range(im_size),range(im_size),indexing='xy')
	x = 2*x/(im_size-1) - 1
	y = 2*y/(im_size-1) - 1
	x *= max_pos_x # scale to match focal length
	y *= max_pos_y
	pix_grid = np.stack([x,-y],0)
	ray_grid = np.concatenate([pix_grid,-np.ones(shape=[1,im_size,im_size])],0)
	ray_grid_flat = ray_grid.reshape(3,-1)
	
	rays = np.matmul(cam_rot,ray_grid_flat)
	rays = rays/np.linalg.norm(rays,2,0)
	rays = rays.reshape(3,im_size,im_size)

	camera_center = np.tile(cam_center[:,None,None],(1,im_size,im_size))
	camera_data = np.concatenate([rays,camera_center],0)

	camera_data = camera_data.astype(np.float32)
	return camera_data

class Scannet_dataset_video(data.Dataset):
	def __init__(self,split):
		super().__init__()
		self.root = f'/cluster/gimli/brossle/scannet/{split}/'
		#self.root = f'/cluster/gimli/brossle/scannet/scans/'
		self.write_samples_for_debugging = False

		self.sequence_codes = list(os.listdir(self.root))

		self.im_size = 128 # used by rel_camera_ray_encoding, size of the loaded images is actually 256

		# check for infinite poses
		self.valid_frames = dict()
		valid_frames_json = os.path.join(os.path.dirname(__file__), "valid_scannet_frames.json")
		if os.path.exists(valid_frames_json):
			with open(valid_frames_json, 'r') as cf:
				self.valid_frames = json.load(cf)
		else:
			print("Checking all poses and exclude frames with infinite poses. Takes a few min.")
			for n, scene in enumerate(self.sequence_codes):
				n_poses = len(os.listdir(os.path.join(self.root, scene, "pose")))
				self.valid_frames[scene] = list()
				for i in range(n_poses):
					if read_pose(self.root, scene, i, verbose=False) is not None:
						self.valid_frames[scene].append(i)
			with open(valid_frames_json, 'w') as f:
				json.dump(self.valid_frames, f, indent=4)
			print("Done. Saved as {}".format(valid_frames_json))

	def __len__(self):
		return len(self.sequence_codes)

	def __getitem__(self,idx):
		sequence_code = self.sequence_codes[idx]
		# idx indicates sequence idx, we will pick 2 random frames from there
		# todo: figure out how to select 2 frames in scannet to have overlap, for now: sample a frame within the +/-20 valid frames
		sample_range = 20
		valid_frame_a_idx = np.random.randint(sample_range, len(self.valid_frames[sequence_code]) - sample_range)
		valid_frame_b_idx = valid_frame_a_idx + np.random.randint(1, sample_range + 1) * (1 if np.random.random() < 0.5 else -1)
		frame_a_idx = self.valid_frames[sequence_code][valid_frame_a_idx]
		frame_b_idx = self.valid_frames[sequence_code][valid_frame_b_idx]

		# load images and intrinsics
		frame_a = read_rgb(self.root, sequence_code, frame_a_idx)
		frame_b = read_rgb(self.root, sequence_code, frame_b_idx)
		K = read_intrinsics(self.root, sequence_code)
		orig_h, orig_w, _ = frame_a.shape
		# marginally crop images to center the prinicipal point to fulfill assumptions made in rel_camera_ray_encoding
		orig_cx, orig_cy = K[0, 2], K[1, 2]
		crop_x, crop_y = round(2 * orig_cx - orig_w), round(2 * orig_cy - orig_h)
		frame_a = crop_image(frame_a, crop_x, crop_y)
		frame_b = crop_image(frame_b, crop_x, crop_y)
		K = crop_intrinsics(K, crop_x, crop_y)

		# ensure images have 360 height
		if frame_a.shape[0] != 360:
			new_w = round(frame_a.shape[1] * (360 / frame_a.shape[0]))
			frame_a = np.asarray(Image.fromarray(frame_a).resize((new_w,360)))
			frame_b = np.asarray(Image.fromarray(frame_b).resize((new_w,360)))
			K = resize_intrinsics(K, new_w / orig_w, 360 / orig_h)

		# crop and downsample
		left_pos = (frame_a.shape[1]-360)//2
		frame_a_cropped = frame_a[:,left_pos:left_pos+360,:]
		frame_b_cropped = frame_b[:,left_pos:left_pos+360,:]
		K = crop_intrinsics(K, left_pos, 0)
		im_a = Image.fromarray(frame_a_cropped).resize((256,256))
		im_b = Image.fromarray(frame_b_cropped).resize((256,256))
		K = resize_intrinsics(K, 256/360, 256/360)
		im_a = np.asarray(im_a).transpose(2,0,1)/127.5 - 1
		im_b = np.asarray(im_b).transpose(2,0,1)/127.5 - 1

		# scale image values
		frame_a_cropped = frame_a_cropped/127.5 - 1
		frame_b_cropped = frame_b_cropped/127.5 - 1

		# convert from scannet (==colmap/opencv) to blender
		tform_a = read_pose(self.root, sequence_code, frame_a_idx)
		tform_b = read_pose(self.root, sequence_code, frame_b_idx)
		tform_a = convert_pose_scannet_to_blender(tform_a)
		tform_b = convert_pose_scannet_to_blender(tform_b)

		tform_a_inv = np.linalg.inv(tform_a)
		tform_b_inv = np.linalg.inv(tform_b)
		tform_ref = np.eye(4)
		tform_a_relative = np.matmul(tform_b_inv,tform_a)
		tform_b_relative = np.matmul(tform_a_inv,tform_b)
		
		# get focal length, create camera ray encoding
		# normalize focal length by image size
		focal_x, focal_y = K[0, 0] / 256, K[1, 1] / 256
		camera_enc_ref = rel_camera_ray_encoding2(tform_ref, self.im_size, focal_x, focal_y)
		camera_enc_a = rel_camera_ray_encoding2(tform_a_relative, self.im_size, focal_x, focal_y)
		camera_enc_b = rel_camera_ray_encoding2(tform_b_relative, self.im_size, focal_x, focal_y)
		
		if self.write_samples_for_debugging:
			visu_dir = os.path.join(os.path.dirname(__file__), "visu")
			os.makedirs(visu_dir, exist_ok=True)
			visu_id = idx%100
			cv2.imwrite(os.path.join(visu_dir, "{}_{}.jpg".format(visu_id, "a")), cv2.cvtColor(((im_a.transpose(1, 2, 0) * 0.5 + 0.5)* 255).clip(0,255).astype(np.uint8), cv2.COLOR_BGR2RGB))
			cv2.imwrite(os.path.join(visu_dir, "{}_{}.jpg".format(visu_id, "b")), cv2.cvtColor(((im_b.transpose(1, 2, 0) * 0.5 + 0.5)* 255).clip(0,255).astype(np.uint8), cv2.COLOR_BGR2RGB))
		
		out_dict = {
			'sequence_code':self.sequence_codes[idx],
			'im_a': im_a.astype(np.float32),
			'im_b': im_b.astype(np.float32),
			'im_a_full': frame_a_cropped.transpose(2,0,1).astype(np.float32),
			'im_b_full': frame_b_cropped.transpose(2,0,1).astype(np.float32),
			'camera_enc_ref': camera_enc_ref,
			'camera_enc_a': camera_enc_a,
			'camera_enc_b': camera_enc_b,
			'tform_ref': tform_ref,
			'tform_a_relative': tform_a_relative,
			'tform_b_relative': tform_b_relative,
			'tform_ref': tform_ref,
			'tform_a': tform_a,
			'tform_b': tform_b,
			'focal_a': focal_y, # does not seem to be used anywhere, setting to focal_y although focal length for x and y is not exactly the same
			'focal_b': focal_y, # does not seem to be used anywhere, setting to focal_y although focal length for x and y is not exactly the same
		}
		return out_dict

