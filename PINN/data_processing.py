def _bound_sample(x_left, x_right, x_updown, y_left, y_right, y_updown, ratio = 4, device = "cpu"):
        perm_x_left = np.random.randint(len(x_left), size=10)     
        perm_y_left = perm_x_left

        x_l_sample =torch.Tensor(x_left[perm_x_left]).to(device)
        y_l_sample = torch.Tensor(y_left[perm_y_left]).to(device)
        # right boudanry sample
        perm_x_right = np.random.randint(len(x_right), size=10)
        perm_y_right = perm_x_right

        x_r_sample = torch.Tensor(x_right[perm_x_right]).to(device)
        y_r_sample = torch.Tensor(y_right[perm_y_right]).to(device)
        
        # up boundary sample
        perm_x_updown = np.random.randint(len(x_updown), size=ratio*10)
        perm_y_updown = perm_x_updown
        x_u_sample = torch.Tensor(x_updown[perm_x_updown]).to(device)
        y_u_sample = torch.Tensor(y_updown[perm_y_updown]).to(device)


        # low boundary sample
        # perm_x_down = np.random.randint(len(x_down), size=ratio*10)
        # perm_y_down = perm_x_down
        # x_d_sample = torch.Tensor(x_down[perm_x_down]).to(device)
        # y_d_sample = torch.Tensor(y_down[perm_y_down]).to(device)
        return x_l_sample, x_r_sample, x_u_sample, y_l_sample, y_r_sample, y_u_sample
def _addnoise(noise_lv, sp_udom, sp_vdom, sp_Pdom):
    for i in range(0,len(sp_udom)):
        u_error = np.random.normal(0, noise_lv*np.abs(sp_udom[i]), 1)

        v_error = np.random.normal(0, noise_lv*np.abs(sp_vdom[i]), 1)
        p_error = np.random.normal(0, noise_lv*np.abs(sp_Pdom[i]), 1)
        sp_udom[i] += u_error
        sp_vdom[i] += v_error
        sp_Pdom[i] += p_error

    return sp_udom, sp_vdom, sp_Pdom
def _paste_b(x_l_sample, x_r_sample, x_u_sample, x_d_sample, y_l_sample,y_r_sample,y_u_sample,y_d_sample, device = "cpu"):

    xb =torch.cat((x_u_sample,x_d_sample),0).to(device)
    xb =torch.cat((x_l_sample,x_r_sample,x_u_sample,x_d_sample),0).to(device)
    yb = torch.cat((y_u_sample,y_d_sample),0).to(device)
    xb = xb.view(len(xb),-1)
    yb = yb.view(len(yb),-1)
    return xb, yb
def _paste_d(sp_xdom, sp_xb, sp_ydom, sp_yb, sp_udom, sp_ub, device = "cpu"):
    ##
    sp_x = np.concatenate((sp_xdom,sp_xb),0)
    sp_y = np.concatenate((sp_ydom,sp_yb),0)
    sp_u = np.concatenate((sp_udom,sp_ub),0)
    # sp_v = np.concatenate((sp_vdom,sp_vb),0)
    # sp_P = np.concatenate((sp_Pdom,sp_Pb),0)
    sp_x, sp_y, sp_u = sp_x[...,None], sp_y[...,None], sp_u[...,None]
    sp_data = np.concatenate((sp_x,sp_y,sp_u),1)


    ##
    # for sparase stenosis
    sp_x, sp_y, sp_u = torch.Tensor(sp_data[:,0]).to(device), torch.Tensor(sp_data[:,1]).to(device), torch.Tensor(sp_data[:,2]).to(device)
    sp_x, sp_y, sp_u = sp_x.view(len(sp_x), -1), sp_y.view(len(sp_y), -1), sp_u.view(len(sp_u), -1)
    return sp_x, sp_y, sp_u
def _paste_in(x_in, y_in, u_in,device = "cpu"):
    x_in = torch.Tensor(x_in).to(device)
    y_in = torch.Tensor(y_in).to(device)
    u_in = torch.Tensor(u_in).to(device)
    x_in, y_in, u_in = x_in.view(len(x_in),-1), y_in.view(len(y_in),-1), u_in.view(len(u_in),-1)
    return x_in, y_in, u_in

def _monitor(loss_1, loss_2, loss_3, loss_d, loss_b, loss_1i, loss_2i, loss_3i, sp_output_i, sp_target, outputb_i):
    loss_f = nn.MSELoss()
    loss_1 += loss_f(loss_1i,torch.zeros_like(loss_1i))
    loss_2 += loss_f(loss_2i,torch.zeros_like(loss_2i))
    loss_3 += loss_f(loss_3i,torch.zeros_like(loss_3i))
    loss_d += loss_f(sp_output_i, sp_target)
    loss_b += loss_f(outputb_i, torch.zeros_like(outputb_i))

    return loss_1, loss_2, loss_3, loss_d, loss_b