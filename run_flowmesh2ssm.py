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
from trainers.mesh2ssm_flow import *
from utils.dataset_meshes import *
from test import test_all

torch.cuda.empty_cache() 
temp = 1000 * 1024 * 1024
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f'max_split_size_mb:{temp}'
parser = argparse.ArgumentParser()


parser.add_argument('--config', type=str, default="configs/la_mesh2ssm3d.json")

args = parser.parse_args()
machine = args.machine
with open(args.config, 'rt') as f:
	t_args = argparse.Namespace()
	t_args.__dict__.update(json.load(f))
	args = parser.parse_args(namespace=t_args)
print(os.path.basename(args.data_dir))
print(os.path.basename(args.log_root))

if(args.seed == None):
	args.seed = np.random.randint(1, 10000)
seed_all(args.seed)


# Logging
if args.logging:
	log_dir = get_new_log_dir(args.log_root, prefix='Flow_', postfix='_' + args.tag if args.tag is not None else '')
	args.folder_name = os.path.basename(log_dir)
	# input(args.folder_name)
	logger = get_logger('train', log_dir)
	writer = torch.utils.tensorboard.SummaryWriter(log_dir)
	ckpt_mgr = CheckpointManager(log_dir)
	log_hyperparams(writer, args)
else:
	logger = get_logger('train', None)
	writer = BlackHole()
	ckpt_mgr = BlackHole()
logger.info(args)

# input("Before loading dataset")
if(args.multi_organ):
	train_dset = MultiOrganDataset(args, partition='train')
	val_dset = MultiOrganDataset(args, partition ='val', size = train_dset.max_size)
	
else:

	train_dset = MeshesWithFaces(args, partition='train')
	val_dset = MeshesWithFaces(args, partition ='val')
	
args.num_vertices = train_dset.max_size
train_iter = DataLoader(
	train_dset,
	batch_size=args.train_batch_size,
	num_workers=8,
	shuffle=True
)

val_iter = DataLoader(
	val_dset,
	batch_size=args.val_batch_size,
	num_workers=1,
	drop_last=False,
	shuffle=True
)

template = np.loadtxt(args.data_dir + "/"+args.template_type)
template = torch.from_numpy(template).type(torch.float)

args.input_x_T = template




# Model
logger.info('Building model...')
model = FlowMesh2SSM(args)
scaled_template = model.input_x_T_scaled
model.set_template(scaled_template)
logger.info(repr(model))
args.model = "mesh2ssm_flow"

# Define your model optimizer choices
optimizer_choices = {
	'adam': torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4),
	'adamw': torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4),
	'sgd': torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4),
}

# Choose an optimizer 
optimizer = optimizer_choices[args.optimizer]


def test():
	# criterion = ChamferDistance()
	corr_chamfer_dist_l1 = []
	corr_chamfer_dist_l2 = []
	# torch.cuda.empty_cache() 
	model.eval()
	for data in val_iter:
		vertices = data['true_pointcloud'].to(args.device)
		n = data['name']
		idx = data['idx'].to(args.device)

		with torch.no_grad():
			correspondences_pred = model.predict(vertices = vertices, idx = idx.to(args.device))


		# Predicted correspondence Chamfer distance 
		val_batch_size = vertices.shape[0]
		
		cd_l1,_ = pytorch3d.loss.chamfer_distance(vertices.reshape((args.val_batch_size,-1,3)), correspondences_pred.reshape((args.val_batch_size,-1,3)), point_reduction='sum', norm=1)
		cd_l2,_ = pytorch3d.loss.chamfer_distance(vertices.reshape((args.val_batch_size,-1,3)), correspondences_pred.reshape((args.val_batch_size,-1,3)), point_reduction='sum', norm=2)
		
		

		corr_chamfer_dist_l1.append(cd_l1.detach().item())
		corr_chamfer_dist_l2.append(cd_l2.detach().item())
	corr_chamfer_dist_l1 = np.array(corr_chamfer_dist_l1)
	corr_chamfer_dist_l2 = np.array(corr_chamfer_dist_l2)

	return np.mean(corr_chamfer_dist_l1), np.mean(corr_chamfer_dist_l2)


		
args.template_update_start = 1500
batch_iter = int(train_iter.__len__())
# Main loop
logger.info('Start training...')
best_val_loss = float('inf')
best_template = None
template_update_counter = 0
autoencoder_burnin = 200
original_consistency_weight = model.args.consistency_weight
try:
	early_stopping_counter = 0
	it = 0
	for epoch in range(args.epochs):
		# torch.cuda.empty_cache() 
		if(epoch<autoencoder_burnin):
			model.args.consistency_weight = 0
		else:
			model.args.consistency_weight = original_consistency_weight

		model.train()
		for batch in train_iter:
			# Reset grad and model state
			optimizer.zero_grad()
			
			
			
			idx = batch['idx'].to(args.device)
			gt_vertices = batch['true_pointcloud'].to(args.device)
			
			
			loss, loss_cd, loss_consistency, loss_dgcnn, loss_flow = model.get_loss_mesh_consistency_perturb(gt_vertices = gt_vertices, idx = idx)
			
			# Backward and optimize
			loss.backward()    
			optimizer.step()
			
			if (it % batch_iter == 0):
				logger.info('[Train] Epoch %04d | Iter %04d | Loss %.6f | Loss CD %.4f | Loss Consistency %.4f | Loss DGCNN %.4f | Loss Flow %.4f' \
					% (epoch, it, loss.mean().item(), loss_cd.mean().item(), loss_consistency.mean().item(), loss_dgcnn.mean().item(), loss_flow.mean().item()))
			it = it +1

			#write outputs
			writer.add_scalar('train/loss_cd', loss_cd.mean().item(), it)
			writer.add_scalar('train/loss_consistency', loss_consistency.mean().item(), it)
			writer.add_scalar('train/loss_dgcnn', loss_dgcnn.mean().item(), it)
			writer.add_scalar('train/loss_flow', loss_flow.mean().item(), it)
			writer.add_scalar('train/loss', loss, it)
			writer.flush()

		# validation loop to plot predicted correspondences and sampled template
		if epoch % args.val_inspect_freq == 0 or epoch == args.epochs:
			
			
			opt_states = {
			'optimizer': optimizer.state_dict(),
			'args': args,               
			'current_epoch': epoch
			}

			# save with the name latest.pt so that you don't waste memory saving all the intermediate models
			ckpt_mgr.save(model, args, 0, others=opt_states, step="latest")
			if(epoch == args.epochs-1):
				filename = log_dir + "/template_" + str(epoch) + ".particles"
				np.savetxt(filename, model.input_x_T.detach().cpu().numpy())


		# valiadation for getting the best model and early stopping check
		if epoch % args.val_freq == 0 or epoch == args.epochs:
			val_loss_cd_l1, val_loss_cd_l2 = test()
			# Early stopping
			logger.info('[Validation] Epoch %04d | Loss CD L1 %.4f | Loss CD L2 %.4f ' \
					% (epoch, val_loss_cd_l1, val_loss_cd_l2))
			if (args.chamfer_dist == 'L1'):
				val_loss = val_loss_cd_l1
			else:
				val_loss = val_loss_cd_l2

			# check for the best model
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				early_stopping_counter = 0
				
				opt_states = {
				'optimizer': optimizer.state_dict(),
				'args': args,
				}

				ckpt_mgr.save(model, args, 0, others=opt_states, step='best_ae')
				filename = log_dir + "/best_template.particles"
				np.savetxt(filename, model.input_x_T.detach().cpu().numpy())
			
			elif epoch>=args.early_stopping_epoch_start:
				early_stopping_counter +=1
				if (early_stopping_counter >= args.early_stopping_patience ):
					logger.info("Early stopping! No improvement in validation loss.")   
					break

		if args.update_template == 1 and epoch >= args.template_update_start:
			if epoch%args.template_update_freq == 0 and template_update_counter<3: 
				new_template = model.update_template()
				filename = log_dir + "/update_template_" + str(epoch) + ".particles"
				np.savetxt(filename, new_template)
				early_stopping_counter = 0
				template_update_counter = template_update_counter + 1
		else:
			print("no template update")

	test_args = args
	test_args.input_x_T = 0
	test_argsname = log_dir + "/test_args.json"
	with open(test_argsname, 'w') as f:
		json.dump(test_args.__dict__, f, indent =2)

except KeyboardInterrupt:
	logger.info('Terminating...')

try:
	test_all(args)
except KeyboardInterrupt:
	logger.info('Terminating...')
