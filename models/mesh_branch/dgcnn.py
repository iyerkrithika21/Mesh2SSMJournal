import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tome import BipartiteWithAttention

def encoding_mask_noise(x, mask_rate=0.3, replace_rate=0.2, mask_token_rate=0.8):
    batch_size, num_features, num_points, k = x.shape
    device = x.device

    # Generate random permutations for all samples in the batch
    perm = torch.stack([torch.randperm(num_points, device=device) for _ in range(batch_size)])
    # Compute the number of nodes to mask for each sample
    num_mask_nodes = max(1, int(np.floor(mask_rate * num_points)))

    # Select mask nodes for each sample using the random permutation
    mask_indices = perm[:, :num_mask_nodes]  # Store mask indices

    # Apply masking to all samples in the batch using advanced indexing
    out_x = x.clone()
    for i in range(batch_size):
        out_x[i, :, mask_indices[i], :] = 0.0  # Set masked nodes to zero

    if replace_rate > 0:
        # Compute the number of noise nodes for each sample
        num_noise_nodes = max(1, int(np.floor(replace_rate * num_mask_nodes)))

        # Select token nodes and noise nodes for each sample using advanced indexing
        token_nodes = mask_indices[:, :max(1, int(np.floor(mask_token_rate * num_mask_nodes)))]
        noise_nodes = mask_indices[:, -num_noise_nodes:]

        # Generate random noise nodes for each sample
        noise_to_be_chosen = torch.randint(0, num_points, (batch_size, num_noise_nodes), device=device)

        # Apply masking to token nodes and noise nodes using advanced indexing
        for i in range(batch_size):
            out_x[i, :, token_nodes[i], :] += 1.0  # Add masking token to token nodes
            out_x[i, :, noise_nodes[i], :] = x[i, :, noise_to_be_chosen[i], :]  # Replace noise nodes with random nodes

    return out_x, mask_indices.tolist()



def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    # print(pairwise_distance.shape)
    idx = pairwise_distance.topk(k=k, dim=-1)[1] # (batch_size, num_points, k)
    return idx



def get_graph_feature(x, k=20, minus_center=True,idx=None):

    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    else:

        idx = idx[:,: ,:k]
    
    batch_size, num_points, _ = idx.size()
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if minus_center:
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    else:
        feature = torch.cat((x, feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class DGCNN_AE(nn.Module):
    def __init__(self, args, flow=False):
        super(DGCNN_AE, self).__init__()
        self.args = args
        self.device = args.device
        self.k = args.k
        # self.masking = args.masking
        # if(self.masking==1):
        #     self.mask_rate = args.mask_rate # ratio of nodes being masked i.e., set to zero
        #     self.replace_rate = args.replace_rate # subset nodes being replaced instead of being of being set to zero
        #     self.mask_token_rate = args.mask_token_rate # subset nodes being replaced with a specific value (1) instead of zero or random noise

        if(flow == True):
            self.emb_dims = args.emb_dims*2
        else:
            self.emb_dims = args.emb_dims

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(self.emb_dims+(64*3), 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 3, kernel_size=1, bias=False)
        


    def forward(self, x,idx=None,v_features=False):

        
        x = x.permute(0, 2, 1) # (batch_size, features, num_points)
        # print(f'shape at input of DGCNN: {x.shape}')
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k,idx=idx)      # (batch_size, 6, num_points) -> (batch_size, 6*2, num_points, k)
        
        # if(self.masking==1):
        #     x, mask_indices = encoding_mask_noise(x, mask_rate=self.mask_rate, replace_rate=self.replace_rate, mask_token_rate=self.mask_token_rate)


        x = self.conv1(x)                       # (batch_size, 6*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        vertex_features = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        
        # if(self.masking==1):
        #     # Create indices for advanced indexing
            
        #     batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, len(mask_indices[0]))
            
        #     # Set the values at mask_indices to zero using advanced indexing
        #     vertex_features[batch_indices, :, mask_indices] = 0


        global_feature = F.adaptive_avg_pool1d(vertex_features, 1)
        
        x = global_feature.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 320, num_points)

        x = self.conv7(x)                       # (batch_size, 320, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 3, num_points)
        x = x.permute(0, 2, 1)
        # print(f'shape at output of DGCNN: {x.shape}')
        
        if(v_features):
            return global_feature.view(batch_size,-1), vertex_features, x
        else:
            return global_feature.view(batch_size, -1), x
   


# DGCNN Autoencoder/VAE class
class DGCNN_AE_VAE(nn.Module):
    def __init__(self, args, flow=False):
        super(DGCNN_AE_VAE, self).__init__()
        
        self.args = args
        self.device = args.device
        self.k = args.k

        
        self.emb_dims_in = args.emb_dims * 2 if flow else args.emb_dims  # Double emb_dims for VAE
        self.emb_dims_out = args.emb_dims
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims_in)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(64, self.emb_dims_in, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(self.emb_dims_out, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 3, kernel_size=1, bias=False)


    def forward(self, x, idx=None, v_features=False):
        x = x.permute(0, 2, 1) # (batch_size, features, num_points)
        
        batch_size, num_points = x.size(0), x.size(2)

        # Encoder
        x = get_graph_feature(x, k=self.k, idx=idx)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.max(dim=-1, keepdim=False)[0]  # Max pooling along k-nearest neighbors

        x = get_graph_feature(x, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x, k=self.k)
        x = self.conv5(x)
        x = x.max(dim=-1, keepdim=False)[0]

        vertex_features = self.conv6(x)  # Bottleneck (emb_dims)
        global_feature = F.adaptive_avg_pool1d(vertex_features, 1)


        # Decoder
        z = global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points)  # Expand z to match the number of points
        x = self.conv7(z)
        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)

        x = x.permute(0, 2, 1)  # Return to original shape (batch_size, num_points, 3)

        if(v_features):
            return global_feature.view(batch_size,-1), vertex_features, x
        else:
            return global_feature.view(batch_size, -1), x


class DGCNN_AE2(nn.Module):
    def __init__(self, args, flow=False):
        super(DGCNN_AE2, self).__init__()
        self.args = args
        self.device = args.device
        self.k = args.k
        
        if(flow == True):
            self.emb_dims = args.emb_dims*2
        else:
            self.emb_dims = args.emb_dims

        # encoder
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)
        

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        # decoder
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.conv6 = nn.Sequential(nn.Conv1d(self.emb_dims, 512, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv8 = nn.Conv1d(256, 3, kernel_size=1, bias=False)
        
    def forward(self, x, idx=None, v_features=False):
        x = x.permute(0, 2, 1) # (batch_size, features, num_points)
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # Graph features 
        x = get_graph_feature(x, k=self.k, idx=idx)
        x = self.conv1(x)
        x1 = F.adaptive_max_pool2d(x, (num_points, 1)).squeeze(-1)
        
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = F.adaptive_max_pool2d(x, (num_points, 1)).squeeze(-1)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = F.adaptive_max_pool2d(x, (num_points, 1)).squeeze(-1)

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = F.adaptive_max_pool2d(x, (num_points, 1)).squeeze(-1)
        

        # Concatenate features from all layers
        x = torch.cat((x1, x2, x3, x4), dim=1)
        vertex_features = self.conv5(x)

        global_feature = F.adaptive_avg_pool1d(vertex_features, 1)

        
        x = global_feature.repeat(1, 1, num_points)  # (batch_size, emb_dims, num_points)

        
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.dp1(x)
        x = self.conv8(x)
        x = x.permute(0, 2, 1)

        if v_features:
            return global_feature, vertex_features, x
        else:
            return global_feature, x