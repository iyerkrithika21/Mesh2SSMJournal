import os
import sys
import glob
import pickle
import numpy as np
from torch.utils.data import Dataset
import pyvista as pv
try:
    from torch_geometric.utils import geodesic_distance
except:
    print("torch_geometric issues")
import torch
from scipy.spatial.distance import pdist, squareform
import time



def geodescis(pos, face,k, max_gdist):
    
    pos= torch.Tensor(pos)
    face = torch.Tensor(face)
    # try:
    dist = -1*geodesic_distance(pos,face.t(),norm=False,num_workers=1, max_distance = max_gdist,)
    idx = dist.topk(k=k,dim=-1)[1]
    # except:
        # print("Geodesics error")
    return idx

def load_meshes_with_faces(directory, partition, extention,k, max_gdist=None, organ=None):

    if(max_gdist==None):
        max_gdist = 5
    
    files = sorted(glob.glob(directory + "/*."+extention))
     
    if(len(files) == 0):
        raise Exception("No files. Check the extention of meshes specified") 
    max_size = 0
    vertices_all = []
    faces_all = []
    pk_filename = directory + 'idx_' + str(k) + "_" +  partition + f'_geodescis_{max_gdist}.pkl'
    try:
        save = False
        with open(pk_filename, 'rb') as f:
            idx_all = pickle.load(f)

    except:
        save = True
        idx_all = {}
    
    max_scale = 0
    filename = []
    for f in files:
        name = f.split("/")[-1]
        if(organ!=None):
            name = f'{organ}/{name}'
        filename.append(name)
        print(f'Loading file: {name}')
        mesh = pv.read(f)
        vertices = np.array(mesh.points).astype('float')
        faces = np.asarray(mesh.faces).reshape((-1, 4))[:, 1:]
        if (save == True or name not in  idx_all.keys() ):
            print("calling geodescis function")
            idx = geodescis(vertices, faces,k, max_gdist)
            idx_all[name] = idx[:,:k]
            with open(pk_filename, 'wb') as f:
                pickle.dump(idx_all,f)
        scale = np.max(np.abs(vertices))
        if(scale>max_scale):
            max_scale = scale
        vertices_all.append(vertices)
        faces_all.append(faces)
        
        if (len(vertices)>max_size):
            max_size = len(vertices)
            print(f'max size of vertices: {max_size}')
        
    if(save ==True):
        with open(pk_filename, 'wb') as f:
            pickle.dump(idx_all,f)
    print("returning")
    return vertices_all, faces_all,idx_all, max_size, max_scale, filename


class MeshesWithFaces(Dataset):
    def __init__(self, args, scale_mode=None, partition = 'train', extention = "ply", noise_level=0,shuffle_points=False,size=None):
        self.mesh_dir = args.data_dir + f'/{partition}/meshes/'
        self.data, self.faces_all, self.idx_all, self.max_size, self.scale, self.filename = load_meshes_with_faces(self.mesh_dir, partition, args.mesh_extension,args.k,args.max_gdist)
        self.partition = partition     
        self.noise_level = noise_level
        self.scale_mode = scale_mode
        self.shuffle_points = shuffle_points
        if(size!=None):
            self.max_size = size


    def __getitem__(self, item):
        name = self.filename[item]
        pointcloud = self.data[item]
        faces = self.faces_all[item]

        excess = self.max_size - len(pointcloud)
        idx = self.idx_all[name]
        idx_extended = idx
        

        list_idx = list(range(len(pointcloud)))
        if(excess > 0):
            repeat_idx = np.random.randint(0,len(pointcloud),excess)
            list_idx = list_idx + list(repeat_idx)

        pc = pointcloud[list_idx,:]
        idx_extended = idx[list_idx,:]
        faces = faces[list_idx, :]
       
        
        pc = torch.from_numpy(pc).type(torch.float)

        # else:
        shift = torch.zeros([1, 3])
        scale = torch.ones([1, 1])

        
        max_value = pc.max()
        
        if(self.noise_level>0):
            noisy_pc = pc + ((self.noise_level*max_value) * np.random.randn(*pc.shape))
        else:
            noisy_pc = pc

        noisy_pc = noisy_pc.type(torch.float)
        pc = pc.type(torch.float)
        
        pc_dict = {
                    'pointcloud': noisy_pc,
                    'name':name,
                    'shift': shift,
                    'scale': scale,
                    'idx': idx_extended,
                    'true_pointcloud': pc,
                    'faces': faces
                }
        return pc_dict
        
    def __len__(self):
        return len(self.data)







class MultiOrganDataset(Dataset):
    def __init__(self, args, partition = 'train', extention = "ply", size=None):
        super(MultiOrganDataset, self).__init__()
        self.args = args
        self.partition = partition
        self.organs = ['L1', 'L2', 'L3', 'L4', 'L5', 'T1', 'T2', 'T3','T4', 'T5','T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
        self.data_dir = args.data_dir
        self.data = []
        self.names  = []
        self.organ_labels = []
        self.idx_all={}
        
        self.max_size = 5000

        for o in self.organs:
            #vertices_all, faces_all,idx_all, max_size, max_scale, filename
            particles,_, idx_all, max_size, scale, names = load_meshes_with_faces(f'{self.data_dir}/{o}/{partition}/meshes/', partition, args.mesh_extension,args.k,args.max_gdist, o)    
            organ_label = [o]*len(names)        

            [self.names.append(n) for n in names]
            [self.organ_labels.append(ol) for ol in organ_label]
            [self.data.append(p) for p in particles]
            # [self.idx_all.append(idx) for idx in idx_all]
            self.idx_all.update(idx_all)

            if(max_size>self.max_size):
                self.max_size = max_size

        if(self.args.tiny_test):
            self.data = self.data[:20]
            self.organ_labels = self.organ_labels[:20]
            self.names = self.names[:20]
            self.idx_all = self.idx_all[:20]
        
       



    def __getitem__(self, item):
        
        pointcloud = self.data[item]
        name = self.names[item]
        organ = self.organ_labels[item]
        


        excess = self.max_size - len(pointcloud)
        idx = self.idx_all[name]
        idx_extended = idx
        

        list_idx = list(range(len(pointcloud)))
        if(excess > 0):
            repeat_idx = np.random.randint(0,len(pointcloud),excess)
            list_idx = list_idx + list(repeat_idx)

        pc = pointcloud[list_idx,:]
        idx_extended = idx[list_idx,:]
        
       
        pc = torch.from_numpy(pc).type(torch.float)


        pc = pc.type(torch.float)
        
        pc_dict = {
                    
                    'name':name,
                    'organ': organ,
                    'idx': idx_extended,
                    'true_pointcloud': pc,
                    
                }
        return pc_dict

        
    def __len__(self):

        return len(self.data)