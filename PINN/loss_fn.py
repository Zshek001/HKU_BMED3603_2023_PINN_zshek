def criterion(model, x,y):

    #print (x)
    #x = torch.Tensor(x).to(device)
    #y = torch.Tensor(y).to(device)
    #t = torch.Tensor(t).to(device)

    #x = torch.FloatTensor(x).to(device)
    #x= torch.from_numpy(x).to(device)
    ntrain = 4500
    x = torch.FloatTensor(x).view(len(x),-1).to(device)
    y = torch.FloatTensor(y).view(len(y),-1).to(device)
    x.requires_grad = True
    y.requires_grad = True




    net_in = torch.cat((x,y),1)

    output = model.forward(net_in)
    u = output[:,0]
    v = output[:,1]
    P = output[:,2]
    u = u.view(len(u),-1)
    v = v.view(len(v),-1)
    P = P.view(len(P),-1)


    # axisymetric
    # R = self.scale * 1/np.sqrt(2*np.pi*self.sigma**2)*torch.exp(-(x-self.mu)**2/(2*self.sigma**2))
    # h = model.rInlet - R

    # u_hard = u*(h**2 - y**2)
    # v_hard = (h**2 -y**2)*v
    u_hard = u
    v_hard = v
    P_hard = (model.xStart-x)*0 + model.dP*(model.xEnd-x)/model.L + 0*y + (model.xStart - x)*(model.xEnd - x)*P
    #P_hard = (-4*x**2+3*x+1)*dP +(xStart - x)*(xEnd - x)*P

    X_scale = 37 #The length of the  domain (need longer length for separation region)
    Y_scale = 4
    U_scale = 10.0
    U_BC_in = 0.1
    Diff = 0.001



    u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

    P_x = torch.autograd.grad(P,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    P_y = torch.autograd.grad(P,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

    #u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(t),create_graph = True,only_inputs=True)[0]
    #v_t = torch.autograd.grad(v,t,grad_outputs=torch.ones_like(t),create_graph = True,only_inputs=True)[0]

    XX_scale = U_scale * (X_scale**2)
    YY_scale = U_scale * (Y_scale**2)
    UU_scale  = U_scale **2

    loss_2 = u*u_x / X_scale + v*u_y / Y_scale - model.nu*( u_xx/XX_scale  + u_yy /YY_scale  )+ 1/model.rho* (P_x / (X_scale*UU_scale))  #X-dir
    loss_1 = u*v_x / X_scale + v*v_y / Y_scale - model.nu*( v_xx/ XX_scale + v_yy / YY_scale )+ 1/model.rho*(P_y / (Y_scale*UU_scale)) #Y-dir
    # loss_1 = (u_hard*u_x+v_hard*u_y-model.nu*(u_xx+u_yy)+1/model.rho*P_x)
    loss_3 = (u_x / X_scale + v_y / Y_scale) #continuity


    # u_x = torch.autograd.grad(u_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    # u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    # u_y = torch.autograd.grad(u_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    # u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    # P_x = torch.autograd.grad(P_hard,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    # #P_xx = torch.autograd.grad(P_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
    # #print('type of nu is',nu.shape)
    # loss_1 = (u_hard*u_x+v_hard*u_y-model.nu*(u_xx+u_yy)+1/model.rho*P_x)

    # v_x = torch.autograd.grad(v_hard,x,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    # v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    
    # v_y = torch.autograd.grad(v_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    
    # v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    # P_y = torch.autograd.grad(P_hard,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
    # #P_yy = torch.autograd.grad(P_y,y,grad_outputs=torch.ones_like(x),create_graph = True,allow_unused = True)[0]


    # loss_2 = (u_hard*v_x+v_hard*v_y - model.nu*(v_xx+v_yy)+1/model.rho*P_y)
    # #Main_deriv = torch.cat((u_x,u_xx,u_y,u_yy,P_x,v_x,v_xx,v_y,v_yy,P_y),1)
    # loss_3 = (u_x + v_y)

    # MSE LOSS
    loss_f = nn.MSELoss()

    #Note our target is zero. It is residual so we use zeros_like
    loss = loss_f(loss_1,torch.zeros_like(loss_1))+  loss_f(loss_2,torch.zeros_like(loss_2))+  loss_f(loss_3,torch.zeros_like(loss_3))




    # logloss1 = ntrain/output.size(0) * (
    #                          0.5 * 1#* (self.nnets[index].log_beta.exp()/ self.nnets[index].log_beta.exp())
    #                         * (loss_1 - torch.zeros_like(loss_1)).pow(2).sum()
    #                         )

    # logloss2 = ntrain/output.size(0) * (
    #                      0.5 * 1#* (self.nnets[index].log_beta.exp()/ self.nnets[index].log_beta.exp())
    #                     * (loss_2 - torch.zeros_like(loss_2)).pow(2).sum()
    #                     )
    # logloss3 = ntrain/output.size(0) * (
    #                      0.5 * 1#* (self.nnets[index].log_beta.exp()/ self.nnets[index].log_beta.exp())
                        # * (loss_3 - torch.zeros_like(loss_3)).pow(2).sum()
                        # )

    # loss = logloss1 + logloss2 + logloss3

    return loss
def Loss_BC(model,xb,yb, inlet_input, outlet_input, inlet_target, outlet_target  ):




    net_in1 = torch.cat((xb, yb), 1)
#     out1_u = net2_u(net_in1)
#     out1_v = net2_v(net_in1)

#     out1_u = out1_u.view(len(out1_u), -1)
#     out1_v = out1_v.view(len(out1_v), -1)

    output = model.forward(net_in1)

    u = output[:,0]
    v = output[:,1]
    
    u = u.view(len(u),-1)
    v = v.view(len(v),-1)



    loss_f = nn.MSELoss()
    loss_noslip = loss_f(u, torch.zeros_like(u))# + loss_f(v, torch.zeros_like(v)) 
    # loss_inlet = loss_f(u, ub_inlet) + loss_f(v, vb_inlet )



    inlet_out = model.forward(inlet_input)
    outlet_out = model.forward(outlet_input)
    loss_ic = loss_f(inlet_out[:,0], inlet_target) + loss_f(outlet_out[:,0], outlet_target)


    return loss_noslip+loss_ic




def Loss_data(model,xd,yd,ud ):


    xb.requires_grad = True
    xd.requires_grad = True
    yd.requires_grad = True


    net_in1 = torch.cat((xd.view(len(xd),-1), yd.view(len(yd),-1)), 1)
    output = model.forward(net_in1)
#     out1_v = net2_v(net_in1)

#     out1_u = out1_u.view(len(out1_u), -1)
#     out1_v = out1_v.view(len(out1_v), -1)
    u = output[:,0]
    v = output[:,1]
    p = output[:,2]
    
    u = u.view(len(u),-1)
    v = v.view(len(v),-1)
    p = p.view(len(p),-1)
    


    loss_f = nn.MSELoss()
    loss_d = loss_f(u, ud) #+ loss_f(v, vd) + loss_f(p, pd) 


    return loss_d