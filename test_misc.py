import os
import math
import argparse
import json
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from utils.misc import *
from trainers.mesh2ssm import *
from trainers.mesh2ssm_flow import *
from utils.dataset_meshes import *
import trimesh
import numpy as np
import pandas as pd
import itertools
import subprocess
from sklearn.decomposition import PCA

def make_directory(sdir):
	if not os.path.exists(sdir):
		os.makedirs(sdir)


torch.cuda.empty_cache() 
temp = 800 * 1024 * 1024
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f'max_split_size_mb:{temp}'

def test_uncertainity(args):
	# Logging
	args.device = 'cuda:0'
	args.attention = 2
	save_dir = args.log_root+ "/" + args.folder_name +"/results/"
	make_directory(save_dir)
	logger = get_logger('test', save_dir)
	for k, v in vars(args).items():
		logger.info('[ARGS::%s] %s' % (k, repr(v)))

	template = np.loadtxt(args.data_dir + "/"+args.template_type)
	template = torch.from_numpy(template).type(torch.float)
	logger.info('Building model...')
	args.input_x_T = template
	
	model = FlowMesh2SSM(args)

	template = np.loadtxt(args.log_root + "/" + args.folder_name + "/best_template.particles")
	scaled_template = torch.from_numpy(template).type(torch.float)
	args.ckpt_path = args.log_root + "/" + args.folder_name + "/ckpt_best_ae_.pt"
	
	checkpoint = torch.load(args.ckpt_path,map_location=args.device)
	
	model.load_state_dict(checkpoint['state_dict'])
	model.set_template(scaled_template)

	model.eval()
	for partition in ['train']:
	
		test_dset = MeshesWithFaces(args, partition = partition)
		test_iter = DataLoader(test_dset,
		batch_size=1,
		num_workers=1,
		drop_last=False
		)
		correspondences_pred_dir = f'{save_dir}/{partition}_correspondences_pred/'
		uncertainty_dir = f'{save_dir}/{partition}_uncertainty/'
		make_directory(correspondences_pred_dir)
		make_directory(uncertainty_dir)
		
		names = []
		corr_p2mDist = []
		corr_particle_list = []
		corr_chamfer_dist_l1 = []
		corr_chamfer_dist_l2 = []
		ale_uncs = []
		projection = False if partition == 'test_partial' else True
		all_zs = []
		with torch.no_grad():
			for data in test_iter:
				vertices = data['true_pointcloud'].to(args.device)
				n = data['name']
				idx = data['idx'].to(args.device)

				# gt_vertices,  idx=None, num_samples=1000
				correspondences_pred, y_log_var, z = model.predict_with_uncertainity(gt_vertices = vertices, idx = idx.to(args.device), num_samples = 100)
				all_zs.append(z.detach().cpu().numpy())
				ale_unc = np.exp(y_log_var.detach().cpu().numpy())
				# save per-particle uncertinity for each sample
				name = n[0].split(".")[0].split("/")[-1] + '.npy'
				np.savetxt(uncertainty_dir + '/' + name , ale_unc.reshape((-1,3)))

				ale_uncs.append(np.sum(ale_unc))
				
				# Predicted correspondence Chamfer distance 
				val_batch_size = vertices.shape[0]
				cd_l1,_ = pytorch3d.loss.chamfer_distance(vertices.reshape((args.val_batch_size,-1,3)), correspondences_pred.reshape((args.val_batch_size,-1,3)),point_reduction='mean', batch_reduction='mean', norm=1)
				cd_l2,_ = pytorch3d.loss.chamfer_distance(vertices.reshape((args.val_batch_size,-1,3)), correspondences_pred.reshape((args.val_batch_size,-1,3)),point_reduction='mean', batch_reduction='mean', norm=2)
				
				print(f'{n}: {cd_l2}')
				corr_chamfer_dist_l1.append(cd_l1.detach().cpu().numpy())
				corr_chamfer_dist_l2.append(cd_l2.detach().cpu().numpy())
				
				
		# latent PCA
		all_zs = np.vstack(all_zs)
		np.save(f'{save_dir}/all_zs.npy', all_zs)

		





def latentPCA(args):
	args.device = 'cuda:0'
	template = np.loadtxt(args.data_dir + "/"+args.template_type)
	template = torch.from_numpy(template).type(torch.float)
	
	args.input_x_T = template
	
	model = FlowMesh2SSM(args)

	template = np.loadtxt(args.log_root + "/" + args.folder_name + "/best_template.particles")
	scaled_template = torch.from_numpy(template).type(torch.float)
	args.ckpt_path = args.log_root + "/" + args.folder_name + "/ckpt_best_ae_.pt"
	
	checkpoint = torch.load(args.ckpt_path,map_location=args.device)
	
	model.load_state_dict(checkpoint['state_dict'])
	model.set_template(scaled_template)

	model.eval()

	save_dir = args.log_root+ "/" + args.folder_name +"/results/"
	all_zs = np.load(f'{save_dir}/all_zs.npy')
	# # PCA 
	pca_walk_dir = f'{save_dir}/pca_correspondences_pred_15/'
	os.makedirs(pca_walk_dir, exist_ok=True)

	# Step 2: Perform PCA
	pca = PCA(n_components=3)  # Choose number of components to retain

	num_samples = all_zs.shape[0]
	all_zs = all_zs.reshape((num_samples, -1))

	pca.fit(all_zs)


	# Step 2: Extract PCA components, mean, and standard deviations
	z_mean = torch.tensor(pca.mean_, device=args.device).float()  # Mean of z
	principal_axes = torch.tensor(pca.components_[:3], device=args.device).float()  # Top 3 principal directions
	std_devs = torch.tensor(pca.singular_values_[:3] / np.sqrt(len(all_zs) - 1), device=args.device).float()  # Standard deviations

	# Step 3: Generate PCA walks for the top 3 components
	num_steps = 3
	steps = torch.linspace(-2, 2, num_steps, device=args.device)  # Walk from -2σ to 2σ

	# Create walks for each principal component
	z_walks = []
	for i in range(2):  # Top 3 modes
		# Generate points along the i-th principal component
		z_walk = torch.stack([z_mean + step * std_devs[i] * principal_axes[i] for step in steps])
		z_walks.append(z_walk)

		# Flatten the walks for input into the network
		z_walks_flat = z_walk.view(-1, args.emb_dims)  # Shape: (3 * num_steps, emb_dims)

		# Get correspondences for all points in all walks
		correspondences = model.decode(z_walks_flat)

		# Reshape correspondences back to (3 modes, num_steps, ...)
		correspondences = correspondences.view(3, -1)
		mode = str(i).zfill(2)

		for j in range(num_steps):
			name = f'pca_mode_{mode}_{steps[j]}'
			print(f'PCA walk: {name}')
			p = correspondences[j, :].detach().cpu().numpy()								
			print(p.shape)
			np.savetxt(f'{pca_walk_dir}/{name}.particles', np.reshape(p,(-1,3)))




if __name__ == '__main__':
	
	arg_list = glob.glob("path/to/all/models/*")
	for arg_dir in arg_list:
		test_argsname = arg_dir + "/test_args.json"
		parser = argparse.ArgumentParser()
		try:
			with open(test_argsname, 'rt') as f:
				t_args = argparse.Namespace()
				t_args.__dict__.update(json.load(f))
				test_args = parser.parse_args(namespace=t_args)
			test_uncertainity(test_args)
			latentPCA(test_args)
		except: 
			print("test_arg not found")
