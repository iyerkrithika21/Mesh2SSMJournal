import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Part of code is from GitHub Repository :-
https://github.com/czq142857/IM-NET-pytorch
"""

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "leakyrelu": nn.LeakyReLU(negative_slope=0.2, inplace=True),

}



class ImNet(nn.Module):
    """ImNet layer pytorch implementation."""

    def __init__(
        self,
        dim=3,
        in_features=32,
        out_features=3,
        nf=64,
        nonlinearity="leakyrelu",
        device=None,
        args=None
    ):
        """Initialization.
        Args:
          dim: int, dimension of input points.
          in_features: int, length of input features (i.e., latent code).
          out_features: number of output features.
          nf: int, width of the second to last layer.
          activation: tf activation op.
          name: str, name of the layer.
        """
        super(ImNet, self).__init__()
        self.dim = dim
        self.args = args
        self.in_features = in_features
        self.dimz = dim + in_features
        self.out_features = out_features
        self.nf = nf
        self.activ = NONLINEARITIES[nonlinearity]
        self.fc0 = nn.Linear(self.dimz, nf * 8)
        self.fc1 = nn.Linear(nf * 8 + self.dimz, nf * 8)
        self.fc2 = nn.Linear(nf * 8 + self.dimz, nf * 4)
        self.fc3 = nn.Linear(nf * 4 + self.dimz, nf * 2)
        self.fc4 = nn.Linear(nf * 2 + self.dimz, nf * 1)
        self.fc5 = nn.Linear(nf * 1, out_features)

        self.device = device
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize the weights of Linear layers 
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                
                # Initialize biases to zeros
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


        
    def set_template(self,args,array=None):
        self.template = array        
        


    def get_template(self):
        return self.template


    def forward(self, z, template=None, weighted_feature=False):
        """Forward method.
        Args:
          x: `[batch_size, dim+in_features]` tensor, inputs to decode.
        Returns:
          output through this layer of shape [batch_size, out_features].
        """
       

        batch_size = len(z)
        template_batch = np.repeat(template[np.newaxis,:,:], batch_size, axis=0)
        # input(template_batch.shape)
        template_batch = torch.from_numpy(template_batch).to(self.device)
        if(weighted_feature):
            zs = z.view(-1, template_batch.shape[1], self.in_features)
        else:
            zs = z.view(-1,1,self.in_features).repeat(1,template_batch.size()[1],1)
        
        x_tmp = torch.cat((template_batch, zs), axis=-1)

        pointz = torch.clone(x_tmp)
        x_tmp = self.activ(self.fc0(x_tmp))
        x_tmp = torch.cat([x_tmp, pointz], dim=-1)
        x_tmp = self.activ(self.fc1(x_tmp))
        x_tmp = torch.cat([x_tmp, pointz], dim=-1)
        x_tmp = self.activ(self.fc2(x_tmp))
        x_tmp = torch.cat([x_tmp, pointz], dim=-1)
        x_tmp = self.activ(self.fc3(x_tmp))
        x_tmp = torch.cat([x_tmp, pointz], dim=-1)
        x_tmp = self.activ(self.fc4(x_tmp))
        displacement = self.fc5(x_tmp)
        return displacement 