import os
import math
import argparse
import json
import torch
import torch.utils.tensorboard
from torch.nn import Module
import pytorch3d
import pytorch3d.loss
import sys
sys.path.append("../")

from models.mesh_branch.dgcnn import *
from models.imnet import * 
from models.flow import *
from torch.optim.lr_scheduler import LambdaLR
from .mesh2ssm import MatchingCosineLoss, differentiable_nearest_projection_batch
from numbers import Number
from torch.autograd import Variable
# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
def truncated_normal(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def sample_diagonal_MultiGauss(mu, log_var, n):
	# reference :
	# http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n

	# Convert z_log_var to std
	std = torch.exp(0.5 * log_var)
	def expand(v):
		if isinstance(v, Number):
			return torch.Tensor([v]).expand(n, 1)
		else:
			return v.expand(n, *v.size())
	if n != 1 :
		mu = expand(mu)
		std = expand(std)
	eps = Variable(std.data.new(std.size()).normal_().to(std.device))
	samples =  mu + eps * std
	samples = samples.reshape((n * mu.shape[1],)+ mu.shape[2:])
	return samples
def MSE(predicted, ground_truth):
	return torch.mean((predicted - ground_truth)**2)

class FlowMesh2SSM(Module):
	"""docstring for FlowMesh2SSM"""
	def __init__(self, args):
		super(FlowMesh2SSM, self).__init__()
		self.args = args
		self.noise_level = args.noise_level
		args.latent_flow_hidden_dim = args.emb_dims

		
		self.dgcnn = DGCNN_AE2(args, flow=True).to(args.device)

		self.flow = build_latent_flow(args).to(args.device)
		
		self.imnet = ImNet(in_features=args.emb_dims, nf=args.nf,device=args.device,args=args).to(args.device)
		com = args.input_x_T.mean(axis=0)
		input_x_T_com = args.input_x_T.add(-com)
		template_max = input_x_T_com.norm(dim=1).max().item()
		self.input_x_T_scaled = input_x_T_com/template_max
		self.imnet.set_template(args,self.input_x_T_scaled.numpy())
		self.matchingloss = MatchingCosineLoss()
		self.template_max = template_max


	def set_template(self,input_x_T):
		self.input_x_T = input_x_T
		self.imnet.set_template(self.args, self.input_x_T.cpu().numpy())

	def update_template(self):
		new_template = self.sample_template()
		new_template = new_template[0,:,:].cpu().detach()
		self.set_template(new_template)
		return new_template

	def sample_template(self, num_estimates = 1000):
		w = torch.randn([num_estimates, self.args.emb_dims]).to(self.args.device)
		# Reverse: z: <- w
		z = self.flow(w, reverse=True).view(num_estimates, -1)
		# mean embedding 
		z_mean = torch.mean(z, axis = 0).reshape((1,-1))
		# mean correspondence using IM-Net
		correspondences_mean = self.imnet(z_mean,self.input_x_T.cpu().numpy())
		
		return correspondences_mean

	def sample(self, w, num_points, truncate_std=1):
		batch_size, _ = w.size()
		
		if truncate_std is not None:
			w = truncated_normal(w, mean=0, std=1, trunc_std=truncate_std)
		# Reverse: z <- w.
		z = self.flow(w, reverse=True).view(batch_size, -1)
		
		# import pdb;pdb.set_trace()
		correspondences = self.imnet(z, self.input_x_T.cpu().numpy())
		return correspondences

	def decode(self, z):
		correspondences = self.imnet(z,self.input_x_T.cpu().numpy())
		return correspondences*self.template_max

	def predict(self, vertices=None, idx=None, projection=True):
		com = vertices.mean(axis=1)
		x_com = com.unsqueeze(1).repeat(1, vertices.shape[1], 1)
		x = vertices.add(-x_com)

		# Scale by max distance of pts to origin
		scale, _ = x.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)
		scaled_x = x/scale

		
		z, _ = self.dgcnn(scaled_x, idx)
		z_mu, z_log_sigma = torch.tensor_split(z, 2, dim=1)
		z = reparameterize_gaussian(mean=z_mu, logvar=z_log_sigma)  # (B, F)

		correspondences = self.imnet(z,self.input_x_T.cpu().detach().numpy())
		correspondences = correspondences*scale
		corr_x_com = com.unsqueeze(1).repeat(1, correspondences.shape[1], 1)
		correspondences = correspondences.add(corr_x_com)
		# import pdb;pdb.set_trace()
		if(projection==True):
			indices = torch.randint(vertices.shape[1], (vertices.shape[0], 2048), device=vertices.device)  # Random indices [B, 2048]
			sampled_vertices = torch.gather(vertices, 1, indices.unsqueeze(-1).expand(-1, -1, vertices.shape[2]))  # Gather points [B, 2048, 3]

			correspondences = differentiable_nearest_projection_batch(correspondences, sampled_vertices, sigma=self.args.sigma)
		return correspondences


	
	def sample_diagonal_MultiGauss(mu, log_var, n):
		# reference :
		# http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n

		# Convert z_log_var to std
		std = torch.exp(0.5 * log_var)
		def expand(v):
			if isinstance(v, Number):
				return torch.Tensor([v]).expand(n, 1)
			else:
				return v.expand(n, *v.size())
		if n != 1 :
			mu = expand(mu)
			std = expand(std)
		eps = Variable(std.data.new(std.size()).normal_().to(std.device))
		samples =  mu + eps * std
		samples = samples.reshape((n * mu.shape[1],)+ mu.shape[2:])
		return samples

	def predict_with_uncertainity(self, gt_vertices, faces=None, idx=None, num_samples=1000):
		batch_size = len(gt_vertices)

		com = gt_vertices.mean(axis=1)
		x_com = com.unsqueeze(1).repeat(1, gt_vertices.shape[1], 1)
		x = gt_vertices.add(-x_com)

		# Scale by max distance of pts to origin
		scale, _ = x.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)
		scaled_x = x/scale

		z, _ = self.dgcnn(scaled_x, idx)
		z_mu, z_log_sigma = torch.tensor_split(z, 2, dim=1)
		
		z = sample_diagonal_MultiGauss(z_mu, z_log_sigma, num_samples) # (B*num_samples, F)
		

		correspondences = self.imnet(z,self.input_x_T.cpu().detach().numpy())
		correspondences = correspondences*scale

		corr_x_com = com.unsqueeze(1).repeat(1, correspondences.shape[1], 1)
		correspondences = correspondences.add(corr_x_com)

		correspondences = correspondences.reshape((num_samples, batch_size, -1, 3))
		
		y_mean = correspondences.mean(0)
		y_log_var = torch.log(correspondences.var(0))


		# import pdb;pdb.set_trace()
		indices = torch.randint(gt_vertices.shape[1], (gt_vertices.shape[0], 2048), device=gt_vertices.device)  # Random indices [B, 2048]
		sampled_vertices = torch.gather(gt_vertices, 1, indices.unsqueeze(-1).expand(-1, -1, gt_vertices.shape[2]))  # Gather points [B, 2048, 3]

		y_mean = differentiable_nearest_projection_batch(y_mean, sampled_vertices, sigma=self.args.sigma)
		return y_mean, y_log_var, z

	def get_loss_mesh_consistency_perturb(self, gt_vertices, faces=None, idx=None):
		batch_size = gt_vertices.shape[0]*2
		

		# Center of mass alignment - set origin to 0,0,0
		com = gt_vertices.mean(axis=1)
		x_com = com.unsqueeze(1).repeat(1, gt_vertices.shape[1], 1)
		x = gt_vertices.add(-x_com)

		# Scale by max distance of pts to origin
		scale, _ = x.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)
		scaled_vertices = x/scale


		# Random decision for adding noise
		apply_noise = torch.rand(1).item() < 0.8  # 80% chance of applying noise

		if apply_noise and self.args.noise_level>0:
			noisy_scaled_vertices = scaled_vertices + self.noise_level * torch.randn_like(scaled_vertices).to(gt_vertices.device)
		else:
			noisy_scaled_vertices = scaled_vertices

		# Random decision for applying vertex masking
		apply_masking = torch.rand(1).item() < 0.8  # 80% chance of applying vertex masking

		if apply_masking and self.args.vertex_masking and self.args.mask_rate>0:
			# Randomize the mask rate 
			mask_rate = abs(torch.randn(1).item() * self.args.mask_rate)

			# Decide number of vertices to mask
			num_mask_vertices = int(mask_rate * noisy_scaled_vertices.shape[1])
			for i in range(batch_size // 2):
				drop_indices = torch.randint(0, noisy_scaled_vertices.shape[1], (num_mask_vertices,))

				# Apply masking by setting selected vertices to zero
				noisy_scaled_vertices[i, drop_indices, :] = 0


		# Double batch
		full_x = torch.cat((scaled_vertices, noisy_scaled_vertices), axis=0)
		full_gt = torch.cat((gt_vertices, gt_vertices), axis=0)
		full_idx = torch.cat([idx, idx], axis=0)
		full_scale = torch.cat([scale, scale], axis=0)

		# get the latent representation of mesh: mu and sigma
		z_mesh, reconstruction = self.dgcnn(full_x, full_idx)
		z_mu, z_log_sigma = torch.tensor_split(z_mesh, 2,dim=1)

		# parameterize for gradient flow
		z_mesh = reparameterize_gaussian(mean=z_mu, logvar=z_log_sigma)
		
		w, delta_log_pw = self.flow(z_mesh.squeeze(-1), torch.zeros([batch_size, 1]).to(self.args.device), reverse=False)
		# flow loss
		# # H[Q(z|X)]
		entropy = gaussian_entropy(logvar=z_log_sigma)      # (B, )
		log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)   # (B, 1)
		log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)
		# Loss
		loss_entropy = -entropy.mean()
		loss_prior = -log_pz.mean()
		loss_flow = self.args.kl_weight*(loss_prior+loss_entropy)


		reconstruction = reconstruction*full_scale
		full_com = torch.cat((com, com), axis=0).unsqueeze(1).repeat(1, reconstruction.shape[1], 1)
		reconstruction = reconstruction.add(full_com)
		
		m_correspondences = self.imnet(z_mesh, self.input_x_T.detach().numpy())
		m_correspondences = m_correspondences*full_scale
		corr_x_com = torch.cat((com, com), axis=0).unsqueeze(1).repeat(1, m_correspondences.shape[1], 1)
		m_correspondences = m_correspondences.add(corr_x_com)
		
		indices = torch.randint(full_gt.shape[1], (full_gt.shape[0], 2048), device=full_gt.device)  # Random indices [B, 2048]
		sampled_vertices = torch.gather(full_gt, 1, indices.unsqueeze(-1).expand(-1, -1, full_gt.shape[2]))  # Gather points [B, 2048, 3]

		m_correspondences = differentiable_nearest_projection_batch(m_correspondences, sampled_vertices, sigma=self.args.sigma)

		if(self.args.mse_weight>0):
			loss_dgcnn = F.mse_loss(full_gt.reshape((batch_size,-1,3)), reconstruction.reshape((batch_size,-1,3)), reduction='none')
			loss_dgcnn = loss_dgcnn.mean(axis = (2,1))
			loss_dgcnn =  loss_dgcnn.mean()
		else:
			loss_dgcnn = torch.zeros(1).to(gt_vertices.device)

		if self.args.chamfer_dist == 'L1':
			loss_cd_mesh, _ =  pytorch3d.loss.chamfer_distance(full_gt.reshape((batch_size,-1,3)), m_correspondences.reshape((batch_size,-1,3)), norm=1, point_reduction='mean',  batch_reduction='mean')
		elif self.args.chamfer_dist == 'L2':
			loss_cd_mesh, _ =  pytorch3d.loss.chamfer_distance(full_gt.reshape((batch_size,-1,3)), m_correspondences.reshape((batch_size,-1,3)), norm=2, point_reduction='mean', batch_reduction='mean')
		
		loss_cd_mesh = loss_cd_mesh.mean()
		
		
		# Get consistency loss
		pred, pred2 = torch.split(m_correspondences, int(batch_size/2))
		embedding1, embedding2 = torch.split(z_mesh, int(batch_size/2))
		recon1, recon2 = torch.split(reconstruction, int(batch_size/2))

		# Contrastive loss (using cosine similarity) + consistency loss for correspondence prediction and vertices reconstruction
		consist_loss = F.mse_loss(pred, pred2) + self.matchingloss(embedding1, embedding2) + F.mse_loss(recon1, recon2)
		loss = loss_flow + loss_cd_mesh + (self.args.consistency_weight*consist_loss) + (self.args.mse_weight*loss_dgcnn)
		return loss, loss_cd_mesh, consist_loss, loss_dgcnn, loss_flow


		