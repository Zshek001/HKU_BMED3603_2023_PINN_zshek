import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize  # noqa: F401
from torch.autograd import Variable
import copy
import math
from numpy.random import default_rng
from torch.nn.parameter import Parameter
from torch.distributions import Gamma
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.ticker as ticker
import sys
import os
import gc
import time

import random
random.seed(20)
class cases:
	def __init__(self,frame, bs):
		self.frame = frame
		self.bs = bs
# 		self.epochs = args.epochs
# 		self.bayes_nn = bayes_nn

	
	def dataloader(self):
		
			train_size = self.bs
			N_y = 30
			L = 1
			xStart = 0
			xEnd = xStart+L
			rInlet = 0.05

			nPt = 100
			unique_x = np.linspace(xStart, xEnd, nPt)
			sigma = 0.1
			scale = 0.005
			mu = 0.5*(xEnd-xStart)
			x_2d = np.tile(unique_x,N_y)
			x = x_2d
			x = np.reshape(x,(len(x),1))
	 		


			lat_data = np.load('./ultra_'+str(self.frame)+'.npz')
			# bc = np.load('./boundary_ultra_part.npz')	 
			# sample_data = np.load('./ultra_sample.npz')
			# xbc=bc['x']
			# ybc=bc['y']
			x = lat_data['x']
			# print(x.shape)
			y = lat_data['y']
			u = lat_data['u']
			
			# print(np.stack([x,y,u],axis=1))
			tem = [tuple(k) for k in np.stack([x,y,u],axis=1)]
			d = np.array(tem, dtype=[('x', float), ('y', float),('u', float)])
			# print(d)	 
			d = np.sort(d, order=('x','y'))	 
			# print(d)	 	
			# sp_xdom = sample_data['x']
			# sp_ydom = sample_data['y']
			# sp_udom = sample_data['u']
			sample = range(5,350,20)#rng.choice(100, size=10, replace=False)
			sp_xdom = []
			sp_ydom = []
			sp_udom = []
			sub_all = []
			step = 2106
			# print(d[0])
			for i in range(1,21):

# 				sub_x = x[int(i*step):int(i*step)+400]	
# 				sub_y = y[int(i*step):int(i*step)+400]
# 				sub_u = u[int(i*step):int(i*step)+400] 
# 	 		# sample_y = rng.choice(len(y)-1, size=18, replace=False)
# 				sp_xdom.append(sub_x[sample])
# 				sp_ydom.append(sub_y[sample])
# 				sp_udom.append(sub_u[sample])
				sub = d[int(i*step):int(i*step)+400][sample]
# 				print(sub.shape)   
				sub_all.append(sub)
			# 	# print(sub.shape)
			# 	# print(sub_x)
			# 	# sub_y = d[int(i*1400):int(i*1400)+300]
			# 	# sub_u = d[2][int(i*1400):int(i*1400)+300] 
	 		# # sample_y = rng.choice(len(y)-1, size=18, replace=False)
			sub_all = np.concatenate(sub_all)
			# sub_all = sub_all[~np.isnan(sub_all[:,2])]
			# # print(sub_all.shape)
			sp_xdom.append([j[0] for j in sub_all])
			sp_ydom.append([j[1] for j in sub_all])
			sp_udom.append([j[2] for j in sub_all])
			# sp_vdom = v[sample]
			# sp_Pdom = P[sample]
			
			sp_xdom = np.array(sp_xdom).reshape(-1)
			sp_ydom = np.array(sp_ydom).reshape(-1)
			sp_udom = np.array(sp_udom).reshape(-1)
			print(sp_udom[~np.isnan(sp_udom)].shape)
# 			sp_ydom = sp_ydom[~np.isnan(sp_udom)]
			sp_udom = np.nan_to_num(sp_udom)
# 			print(sp_udom.shape)
			x = x[~np.isnan(u)]
			y = y[~np.isnan(u)]
			u = u[~np.isnan(u)]
# 			# print(sp_xdom)
# 			# print(sp_ydom)
# 			# print(type(sp_xdom[0]))
# 			# sp_vdom = sp_vdom.reshape(-1)
# 			# sp_Pdom = sp_Pdom.reshape(-1)
			np.savez('xyu_sparse_ultra_'+str(self.frame),xdom = sp_xdom,ydom = sp_ydom,udom = sp_udom )#, vdom = sp_vdom, pdom = sp_Pdom, xinlet=xinlet, yinlet=yinlet, uinlet=uinlet, xoutlet=xoutlet, youtlet=youtlet, uoutlet=uoutlet, xb=xb, yb=yb,ub=ub,vb=vb, pb=pb)
			# plt.figure()
			# plt.subplot(2,1,1)
			# plt.scatter(x, y, c= u_hard, label = 'uhard', cmap = 'coolwarm', vmin = min(u_CFD), vmax = max(u_CFD))
			

			R = scale * 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
			nu = 1e-3
			yUp = rInlet - R
			yDown = -rInlet + R
			plt.figure(figsize=(24,6))
			# plt.scatter(x, yUp)
			# plt.scatter(x, yDown)
			plt.scatter(x,y, c=u, cmap='coolwarm')
			# plt.scatter(cor[:,0],cor[:,1],cmap='coolwarm')
			plt.colorbar()
			print(sp_xdom.shape)
			plt.scatter(sp_xdom,sp_ydom, marker='x', c='black')
			plt.axis('equal')
			# plt.scatter(xbc, ybc, marker='x', c='black')
			
			plt.show()
			############################
			# x = np.repeat(np.round(np.linspace(35.242, 39.376, 300),3),10)
			# y = np.round(np.linspace(-2,2,3000),3)
			# np.random.shuffle(y)
			np.savez('stenosis_hard_coord_ultra_'+str(self.frame),x = x,y = y,yUp = yUp,u = u) #, v = v, P = P)
			################
			data = torch.utils.data.TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y))
			
			train_loader = torch.utils.data.DataLoader(data, batch_size=train_size, shuffle=True)
			print('len(data is)', len(data))
			print('len(dataloader is)', len(train_loader))
			return train_loader,train_size
		else:
			raise Exception("error,no such model") 