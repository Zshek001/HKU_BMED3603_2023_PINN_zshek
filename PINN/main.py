
# scitific cal
import numpy as np
from scipy.spatial.distance import pdist, squareform
import copy
import math
# plotting
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
# system
from time import time
import sys
import os
import gc
import subprocess # Call the command line
from subprocess import call
import pdb
# torch import
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim


from args import args
import loss_fn
import data_processing
import bloodnet


train_case  = cases(args.frame, args.batch_size)
train_loader, train_size = train_case.dataloader()
model = FloodNet(2,100)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.apply(init_normal)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.1)
tic = time.time()
Data_sparse = np.load('xyu_sparse_ultra_'+str(args.frame)+'.npz')
sp_xdom = Data['xdom']
sp_ydom = Data['ydom']
sp_udom = Data['udom']
# sp_xin, sp_yin, sp_uin, sp_xout, sp_yout, sp_uout, ,sp_vdom,sp_Pdom,sp_xb,sp_yb,sp_ub,sp_vb,sp_Pb
Data_b = np.load('boundary_ultra_'+str(args.frame)+'.npz')
x_left, x_right, x_updown = Data_b['inlet'][:,0], Data_b['outlet'][:,0], Data_b['x']
y_left, y_right, y_updown = Data_b['inlet'][:,1], Data_b['outlet'][:,1], Data_b['y']
# gra_res = []
# gra_data = []
# gra_bc = []
epo = args.epochs
bc_loss = np.zeros(epo)
res_loss = np.zeros(epo)
data_loss = np.zeros(epo)
total_loss = np.zeros(epo)

for epoch in range(epo):
    #for batch_idx, (x_in,y_in) in enumerate(dataloader):  
    #for batch_idx, (x_in,y_in,xb_in,yb_in,ub_in,vb_in) in enumerate(dataloader): 
    loss_eqn_tot = 0.
    loss_bc_tot = 0.
    loss_data_tot = 0.
    loss_tot = 0
    n = 0
    
    for batch_idx, (x,y) in enumerate(train_loader): 

        model.zero_grad()
        x_l_sample, x_r_sample, x_u_sample, y_l_sample, y_r_sample, y_u_sample = _bound_sample(x_left, x_right, x_updown, y_left, y_right, y_updown)
        xb, yb = x_u_sample.view(len(x_u_sample),-1), y_u_sample.view(len(y_u_sample),-1) #_paste_b(x_l_sample, x_r_sample, x_u_sample, x_d_sample, y_l_sample,y_r_sample,y_u_sample,y_d_sample)
        xb.requires_grad = True
        yb.requires_grad = True
        net_inb = torch.cat((xb,yb),1)
        sp_x, sp_y, sp_u = torch.Tensor(sp_xdom).to(device),torch.Tensor(sp_ydom).to(device),torch.Tensor(sp_udom).to(device)#_paste_d(sp_xdom, sp_xb, sp_ydom, sp_yb, sp_udom, sp_ub)

        sp_x.reuqires_grad = True
        sp_y.requires_grad = True
        sp_inputs = torch.cat((sp_x.view(len(sp_x),-1),sp_y.view(len(sp_y),-1)),1)
        sp_target = sp_u#torch.cat((sp_u,sp_v,sp_P),1)
        ## paste inlet
        x_in, y_in, u_in = torch.Tensor(x_left).to(device), torch.Tensor(y_left).to(device), torch.Tensor(Data_b['inlet'][:,2]).to(device)
        x_in.requires_grad = True
        y_in.requires_grad = True
        inlet_input = torch.cat((x_in.view(len(x_in),-1),y_in.view(len(y_in),-1)),1)
        inlet_target = u_in
        ## paste outlet
        x_out, y_out, u_out = torch.Tensor(x_right).to(device), torch.Tensor(y_right).to(device), torch.Tensor(Data_b['outlet'][:,2]).to(device)
        x_out.requires_grad = True
        y_out.requires_grad = True
        outlet_input = torch.cat((x_out.view(len(x_out),-1), y_out.view(len(y_out),-1)), 1)
        outlet_target = u_out

        loss_eqn = criterion(model,x,y)
        loss_bc = Loss_BC(model, xb,yb, inlet_input, outlet_input, inlet_target, outlet_target )
        loss_data = Loss_data(model, sp_x, sp_y, sp_u.reshape(-1,1))


        loss = loss_eqn +  loss_bc + loss_data
        loss.backward()
        optimizer.step() 
  
        loss_eqn_tot += loss_eqn
        loss_bc_tot += loss_bc
        loss_data_tot  += loss_data
        loss_tot += loss
    
    if epoch % 100 ==0:
          Data = np.load('stenosis_hard_coord_ultra_'+str(args.frame)+'.npz')
          x = Data['x']
          y = Data['y']
          u_CFD = Data['u']
          
          yUp = Data['yUp']
          xt,yt = torch.Tensor(x), torch.Tensor(y)
          Rt= torch.Tensor(yUp).to(device)
          xt,yt = xt.view(len(xt),-1), yt.view(len(yt),-1)
          xt.requires_grad = True
          yt.requires_grad = True
          xt, yt = xt.to(device), yt.to(device)
          inputs = torch.cat((xt,yt),1)
          with torch.no_grad():
              out_all = model.forward(inputs)
              u_hard = out_all[:,0]
              v_hard = out_all[:,1]
              P_hard = out_all[:,2]
              u_hard = u_hard.view(len(u_hard),-1)
              v_hard = v_hard.view(len(v_hard),-1)
              P_hard = P_hard.view(len(P_hard),-1)
              u_hard = u_hard.cpu().detach().numpy()
              v_hard = v_hard.cpu().detach().numpy()
              P_hard = P_hard.cpu().detach().numpy()
          

          plot_x= 0.4*np.max(x)
          plot_y = 0.95*np.max(y)
          fontsize = 18
          

          plt.figure()
          # plt.subplot(2,1,1)
          plt.scatter(x, y, c= u_hard, label = 'uhard', cmap = 'coolwarm', vmin = min(u_CFD), vmax = max(u_CFD))
          plt.text(plot_x, plot_y, r'epoch {}'.format(epoch), {'color': 'b', 'fontsize': fontsize})
          plt.colorbar()
          #plt.axis('equal')
          plt.savefig('../ultra{}/{}.png'.format(args.frame, epoch),bbox_inches = 'tight')
    loss_eqn_tot = loss_eqn_tot
    loss_bc_tot = loss_bc_tot
    loss_data_tot = loss_data_tot
    loss_tot = loss_tot#/n
    bc_loss[epoch] = loss_bc_tot.cpu().detach().numpy()
    res_loss[epoch] = loss_eqn_tot.cpu().detach().numpy()
    data_loss[epoch] = loss_data_tot.cpu().detach().numpy()
    total_loss[epoch] = loss_tot.cpu().detach().numpy()
    print('*****Train Epoch: {} Total avg Loss {:.10f} Loss eqn {:.10f} Loss BC {:.10f} Loss data {:.10f} ****'.format(epoch, loss_tot, loss_eqn_tot, loss_bc_tot,loss_data_tot) )
    # print('learning rate is ', optimizer.param_groups[0]['lr'])
torch.save(model.state_dict(), 'ultra_'+str(args.frame)+'.pt')
print('finished in ', (time.time()-tic))
plt.figure()
plt.plot(range(len(bc_loss)),bc_loss,label='bc_loss')
plt.plot(range(len(res_loss)),res_loss,label='res_loss')
plt.plot(range(len(data_loss)),data_loss,label='data_loss')
plt.plot(range(len(total_loss)),total_loss,label='total_loss')
plt.legend()
plt.savefig('./loss_ultra_'+str(args.frame)+'.png'.format(epoch),bbox_inches = 'tight')
np.savez('train_loss_ultra_'+str(args.frame)+'.npz', bc_loss=bc_loss, res_loss=res_loss, data_loss=data_loss, total_loss=total_loss)