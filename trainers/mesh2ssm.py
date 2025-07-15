import os
import math
import argparse
import json
import torch
import torch.utils.tensorboard
import torch.nn.functional as F

from torch.nn import Module
import pytorch3d
import pytorch3d.loss
import sys
sys.path.append("../")
from models.mesh_branch.dgcnn import *

from models.imnet import * 

def differentiable_nearest_projection_batch(landmarks, mesh_vertices, sigma=0.1):
	"""
	Differentiable nearest projection with batch support.

	Args:
		landmarks (torch.Tensor): Tensor of shape (B, N, 3), batch of N landmarks in B samples.
		mesh_vertices (torch.Tensor): Tensor of shape (B, M, 3), batch of M mesh vertices in B samples.
		sigma (float): Scaling factor for softmin.

	Returns:
		torch.Tensor: Projected landmarks of shape (B, N, 3), 
					  adjusted based on direction to mesh vertices.
	"""
	B, N, _ = landmarks.shape
	_, M, _ = mesh_vertices.shape

	# Compute pairwise distances: (B, N, M)
	distances = torch.cdist(landmarks, mesh_vertices)  # Batch-wise pairwise distances

	# Compute softmin weights: (B, N, M)
	weights = F.softmax(-distances / sigma, dim=-1)

	# Compute closest mesh vertex directions: (B, N, M, 3)
	directions = mesh_vertices.unsqueeze(1) - landmarks.unsqueeze(2)  # (B, N, M, 3)

	# Weighted sum of direction vectors: (B, N, 3)
	displacement = torch.sum(weights.unsqueeze(-1) * directions, dim=2)

	# Update landmarks along the direction vectors
	projected_landmarks = landmarks + displacement

	return projected_landmarks


class MatchingCosineLoss(nn.Module):
	def __init__(self, margin=0.5):
		super(MatchingCosineLoss, self).__init__()
		'''
		Purpose of the Margin: 
		We want non-matching pairs to have a similarity below the margin, ideally closer to zero. 
		When the similarity of a non-matching pair is below the margin, it’s already sufficiently "dissimilar," 
		so we don’t want it to contribute to the loss. This way, the model focuses on separating only 
		those non-matching pairs that are too similar.
		'''
		self.margin = margin

	def forward(self, em1, em2):
		em1 = em1.squeeze(-1)
		em2 = em2.squeeze(-1)
		# Step 1: Normalize embeddings for cosine similarity
		em1_norm = F.normalize(em1, p=2, dim=1)
		em2_norm = F.normalize(em2, p=2, dim=1)

		# Step 2: Compute cosine similarity matrix for all pairs
		cosine_sim_matrix = torch.mm(em1_norm, em2_norm.t())  # Shape: (batch_size, batch_size)

		# Step 3: Positive Loss (same-sample pairs along the diagonal)
		positive_loss = 1 - cosine_sim_matrix.diag().mean()  # maximize diagonal similarity

		# Step 4: Negative Loss (different-sample pairs, off-diagonal)
		batch_size = em1.size(0)
		negative_mask = torch.ones(batch_size, batch_size, device=em1.device) - torch.eye(batch_size, device=em1.device)
		negative_similarity = cosine_sim_matrix * negative_mask
		
		# Apply margin and clamp to focus on pairs with high similarity
		negative_loss = torch.clamp(negative_similarity - self.margin, min=0).mean()

		# Combine positive and negative losses
		loss = positive_loss + negative_loss
		return loss




def MSE(predicted, ground_truth):
	return torch.mean((predicted - ground_truth)**2)

class Mesh2SSM(Module):
	"""docstring for Mesh2SSM"""
	def __init__(self, args):
		super(Mesh2SSM, self).__init__()
		self.args = args
		self.noise_level = args.noise_level

		# if self.args.attention == 1:
		# 	# old attention
		# 	self.dgcnn = DGCNN_AE1_Attention(args).to(args.device)
		# elif self.args.attention == 2:
		# 	self.dgcnn = DGCNN_AE2_Attention(args).to(args.device)
		# else:
		self.dgcnn = DGCNN_AE2(args).to(args.device)
		self.imnet = ImNet(in_features=args.emb_dims, nf=args.nf,device=args.device,args=args).to(args.device)
		
		
		com = args.input_x_T.mean(axis=0)
		input_x_T_com = args.input_x_T.add(-com)
		template_max = input_x_T_com.norm(dim=1).max().item()
		self.input_x_T_scaled = input_x_T_com/template_max
		self.imnet.set_template(args,self.input_x_T_scaled.numpy())
		self.matchingloss = MatchingCosineLoss()



	def set_template(self,input_x_T):
		self.input_x_T = input_x_T

		self.imnet.set_template(self.args, self.input_x_T.numpy())


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

		z_mesh, reconstruction = self.dgcnn(full_x, full_idx)
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
			loss_cd_mesh, _ =  pytorch3d.loss.chamfer_distance(full_gt.reshape((batch_size,-1,3)), m_correspondences.reshape((batch_size,-1,3)), norm=1, point_reduction='sum')#, batch_reduction='mean')
		elif self.args.chamfer_dist == 'L2':
			loss_cd_mesh, _ =  pytorch3d.loss.chamfer_distance(full_gt.reshape((batch_size,-1,3)), m_correspondences.reshape((batch_size,-1,3)), norm=2, point_reduction='sum')#, batch_reduction='mean')
		
		loss_cd_mesh = loss_cd_mesh.mean()
		
		
		# Get consistency loss
		pred, pred2 = torch.split(m_correspondences, int(batch_size/2))
		embedding1, embedding2 = torch.split(z_mesh, int(batch_size/2))
		recon1, recon2 = torch.split(reconstruction, int(batch_size/2))

		# Contrastive loss (using cosine similarity) + consistency loss for correspondence prediction and vertices reconstruction
		consist_loss = F.mse_loss(pred, pred2) + self.matchingloss(embedding1, embedding2) + F.mse_loss(recon1, recon2)
		loss =  loss_cd_mesh + (self.args.consistency_weight*consist_loss) + (self.args.mse_weight*loss_dgcnn)
		return loss, loss_cd_mesh, consist_loss, loss_dgcnn

	def forward(self, vertices=None, idx=None):
		
		com = vertices.mean(axis=1)
		x_com = com.unsqueeze(1).repeat(1, vertices.shape[1], 1)
		x = vertices.add(-x_com)

		# Scale by max distance of pts to origin
		scale, _ = x.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)
		scaled_x = x/scale

		

		z, _ = self.dgcnn(scaled_x, idx)
		correspondences = self.imnet(z,self.input_x_T.cpu().detach().numpy())
		correspondences = correspondences*scale
		corr_x_com = com.unsqueeze(1).repeat(1, correspondences.shape[1], 1)
		correspondences = correspondences.add(corr_x_com)
		
		indices = torch.randint(vertices.shape[1], (vertices.shape[0], 2048), device=vertices.device)  # Random indices [B, 2048]
		sampled_vertices = torch.gather(vertices, 1, indices.unsqueeze(-1).expand(-1, -1, vertices.shape[2]))  # Gather points [B, 2048, 3]

		correspondences = differentiable_nearest_projection_batch(correspondences, sampled_vertices, sigma=self.args.sigma)

		return correspondences

	def predict(self, vertices=None, idx=None, projection=True):
		
		com = vertices.mean(axis=1)
		x_com = com.unsqueeze(1).repeat(1, vertices.shape[1], 1)
		x = vertices.add(-x_com)

		# Scale by max distance of pts to origin
		scale, _ = x.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)
		scaled_x = x/scale

		

		z, _ = self.dgcnn(scaled_x, idx)
		correspondences = self.imnet(z,self.input_x_T.cpu().detach().numpy())
		correspondences = correspondences*scale
		corr_x_com = com.unsqueeze(1).repeat(1, correspondences.shape[1], 1)
		correspondences = correspondences.add(corr_x_com)
		if(projection == True):
			indices = torch.randint(vertices.shape[1], (vertices.shape[0], 1024), device=vertices.device)  # Random indices [B, 2048]
			sampled_vertices = torch.gather(vertices, 1, indices.unsqueeze(-1).expand(-1, -1, vertices.shape[2]))  # Gather points [B, 2048, 3]
			
			
			correspondences = differentiable_nearest_projection_batch(correspondences, sampled_vertices, sigma=2*self.args.sigma)
		
		return correspondences