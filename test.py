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
# from torch_cluster import knn as t_knn
import trimesh
import numpy as np
import pandas as pd
import itertools
import subprocess


"""
Part of code is from GitHub Repository - https://github.com/jadie1/Point2SSM_2
"""
# Get numpy mesh verts and particles
def get_particles(particle_dir):
	particles = []
	
	for file in sorted(os.listdir(particle_dir)):
		if not os.path.isdir(particle_dir+file):
			print(particle_dir+file)
			particles.append(np.loadtxt(particle_dir+file))
		
	return np.array(particles)

def knn(x, k):
	inner = -2 * torch.matmul(x, x.transpose(1, 0).contiguous(), )
	xx = torch.sum(x ** 2, dim=1, keepdim=True)
	pairwise_distance = -xx - inner - xx.transpose(1, 0).contiguous()
	idx = pairwise_distance.topk(k=k, dim=-1) # (batch_size, num_points, k)
	return idx


# all combos within batch
def get_neighbor_loss(pred, k, eval_dir, args):
	pred = torch.FloatTensor(pred).to(args.device)
	edge_index = [knn(pred[i], k,) for i in range(pred.shape[0])]
	neigh_idxs = torch.stack([edge_index[i][1].reshape(pred.shape[1], -1) for i in range(pred.shape[0])])
	batch_size = pred.shape[0]
	loss, count = 0, 0
	losses = []
	for source_index in range(batch_size):
		for target_index in range(batch_size):
			if source_index != target_index:
				count += 1
				losses.append(neighbor_loss_helper(pred[source_index].unsqueeze(0), neigh_idxs[source_index].unsqueeze(0), pred[target_index].unsqueeze(0), k))
	MEs = torch.Tensor(losses)
	np.save(eval_dir + 'me.npy', MEs.cpu().numpy())
	return round(MEs.mean().item(),2), round(MEs.std().item(),2)

def neighbor_loss_helper(source, source_neighs, target, k):
	source_grouped = pytorch3d.ops.knn_gather(source.contiguous(), source_neighs)
	source_diff = source_grouped[:, :, 1:, :] - torch.unsqueeze(source, 2)  # remove fist grouped element, as it is the seed point itself
	source_square = torch.sum(source_diff ** 2, dim=-1)

	target_cr_grouped = pytorch3d.ops.knn_gather(target.contiguous(), source_neighs)
	target_cr_diff = target_cr_grouped[:, :, 1:, :] - torch.unsqueeze(target, 2)  # remove fist grouped element, as it is the seed point itself
	target_cr_square = torch.sum(target_cr_diff ** 2, dim=-1)

	GAUSSIAN_HEAT_KERNEL_T = 8.0
	gaussian_heat_kernel = torch.exp(-source_square/GAUSSIAN_HEAT_KERNEL_T)
	neighbor_loss_per_neigh = torch.mul(gaussian_heat_kernel, target_cr_square)

	neighbor_loss = torch.mean(neighbor_loss_per_neigh)
	return neighbor_loss


def make_directory(sdir):
	if not os.path.exists(sdir):
		os.makedirs(sdir)

def write_particles(particle_dir, names, particle_list):
	output_file = []
	for n,p in zip(names, particle_list):
		n = n.split(".")[0].split("/")[-1] + ".particles"
		# print(n)
		p = p.detach().cpu().numpy()
		np.savetxt(particle_dir + n, np.reshape(p,(-1,3)))
		output_file.append(particle_dir + n)
	return output_file

def calculate_point_to_mesh_distance(m,p):
	mesh = trimesh.load(m)
	points = np.loadtxt(p)

	c = trimesh.proximity.ProximityQuery(mesh)
	p2mDist = c.signed_distance(points)

	return p2mDist


torch.cuda.empty_cache() 
temp = 800 * 1024 * 1024
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f'max_split_size_mb:{temp}'
parser = argparse.ArgumentParser()

def test_all(args):
	# Logging
	# args.device = 'cuda:0'
	save_dir = args.log_root+ "/" + args.folder_name +"/results/"
	make_directory(save_dir)
	logger = get_logger('test', save_dir)
	for k, v in vars(args).items():
		logger.info('[ARGS::%s] %s' % (k, repr(v)))

	template = np.loadtxt(args.data_dir + "/"+args.template_type)
	template = torch.from_numpy(template).type(torch.float)
	logger.info('Building model...')
	args.input_x_T = template
	if(args.model == 'mesh2ssm' or args.model=="vanilla_mesh2ssm"):
		model = Mesh2SSM(args)
		
	elif(args.model == 'mesh2ssm_flow'):
		model = FlowMesh2SSM(args)

	template = np.loadtxt(args.log_root + "/" + args.folder_name + "/best_template.particles")
	scaled_template = torch.from_numpy(template).type(torch.float)
	args.ckpt_path = args.log_root + "/" + args.folder_name + "/ckpt_best_ae_.pt"
	
	checkpoint = torch.load(args.ckpt_path,map_location=args.device)
	
	model.load_state_dict(checkpoint['state_dict'])
	model.set_template(scaled_template)

	model.eval()
	if(args.model == 'vanilla_mesh2ssm'):
		projection = False
	else:
		projection = True
	for partition in ['test']:
	
		test_dset = MeshesWithFaces(args, partition = partition)
		test_iter = DataLoader(test_dset,
		batch_size=1,
		num_workers=1,
		drop_last=False
		)
		correspondences_pred_dir = f'{save_dir}/{partition}_correspondences_pred/'
		make_directory(correspondences_pred_dir)
		
		names = []
		corr_p2mDist = []
		corr_particle_list = []
		corr_chamfer_dist_l1 = []
		corr_chamfer_dist_l2 = []
		with torch.no_grad():
			for data in test_iter:
				vertices = data['true_pointcloud'].to(args.device)
				n = data['name']
				idx = data['idx'].to(args.device)

				
				correspondences_pred = model.predict(vertices = vertices, idx = idx.to(args.device), projection=projection)


				# Predicted correspondence Chamfer distance 
				val_batch_size = vertices.shape[0]
				
				cd_l1,_ = pytorch3d.loss.chamfer_distance(vertices.reshape((args.val_batch_size,-1,3)), correspondences_pred.reshape((args.val_batch_size,-1,3)),point_reduction='mean', batch_reduction='mean', norm=1)
				cd_l2,_ = pytorch3d.loss.chamfer_distance(vertices.reshape((args.val_batch_size,-1,3)), correspondences_pred.reshape((args.val_batch_size,-1,3)),point_reduction='mean', batch_reduction='mean', norm=2)
				
				print(f'{n}: {cd_l2}')
				corr_chamfer_dist_l1.append(cd_l1.detach().cpu().numpy())
				corr_chamfer_dist_l2.append(cd_l2.detach().cpu().numpy())
				
				corr_particle = write_particles(correspondences_pred_dir, n, correspondences_pred)				
				corr_particle_list.append(corr_particle)

				#point to surface distance 
				for m,p in zip(n, corr_particle):
					#Correspondence
					m = args.data_dir + f'/{partition}/meshes/{m}'
					names.append(m)
					p2m = calculate_point_to_mesh_distance(m,p)
					corr_p2mDist.append(p2m)
		pred = get_particles(correspondences_pred_dir)
		me_loss_mean, me_loss_std = get_neighbor_loss(pred, 10, f'{save_dir}/{partition}_', args)
		logger.info('Correspondence mapping error: %.6f  +/- %.6f ' % (me_loss_mean,me_loss_std))	
		corr_chamfer_dist_l1 = np.array(corr_chamfer_dist_l1).flatten()
		corr_chamfer_dist_l2 = np.array(corr_chamfer_dist_l2).flatten()
		corr_p2mDist = np.array(corr_p2mDist)
		corr_p2mDist_mean = np.mean(corr_p2mDist,axis=1)
		worst_index, best_index = np.argmax(corr_p2mDist_mean), np.argmin(corr_p2mDist_mean)
		median_index = np.argsort(corr_p2mDist_mean)[len(corr_p2mDist_mean)//2]
		labels = ['worst', 'median', 'best']
		indices = [worst_index, median_index, best_index]
		
		corr_particle_list = list(itertools.chain.from_iterable(corr_particle_list))
		files_dict = {'worst':[corr_particle_list[worst_index],worst_index], 'median':[corr_particle_list[median_index],median_index], 'best':[corr_particle_list[best_index],best_index]}
		pd.DataFrame.from_dict(files_dict).to_csv(f'{save_dir}/{partition}_p2m_file_list.csv', index= False)

		project_dict = {'meshes':names, 'corr_particles':corr_particle_list}	
		print(f'meshes: {len(names)}, corr_particles : {len(corr_particle_list)} ')	
		pd.DataFrame.from_dict(project_dict).to_csv(save_dir + partition + "_file_lists.csv",index=False)
		np.save(f'{save_dir}/{partition}_cd_l1.npy', corr_chamfer_dist_l1)
		np.save(f'{save_dir}/{partition}_cd_l2.npy', corr_chamfer_dist_l2)
		np.save(f'{save_dir}/{partition}_p2mdist.npy', corr_p2mDist)
		
		logger.info('Correspondence Chamfer distance L1: %.6f  +/- %.6f ' % (np.mean(corr_chamfer_dist_l1), np.std(corr_chamfer_dist_l1)))
		logger.info('Correspondence Chamfer distance L2: %.6f  +/- %.6f ' % (np.mean(corr_chamfer_dist_l2), np.std(corr_chamfer_dist_l2)))
		logger.info('Correspondence point to mesh distance: %.6f  +/- %.6f ' % (np.mean(corr_p2mDist), np.std(corr_p2mDist)))

		del logger
