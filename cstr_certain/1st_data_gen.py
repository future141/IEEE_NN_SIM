import torch
import cst
import multiprocessing as mp

# system with control
def sys(x):
    xp = torch.zeros(2,cst.u_step*cst.x_step*cst.x_step)
    xp[0,:] = x[0,:] + cst.del_t*(-x[0,:]              + cst.Da1*      (1-x[0,:])*torch.exp(x[1,:]/(1+x[1,:]/cst.gamma)) + (1-cst.zeta)*x[0,:])
    xp[1,:] = x[1,:] + cst.del_t*(-x[1,:]*(1+cst.beta) + cst.Da2*cst.B*(1-x[0,:])*torch.exp(x[1,:]/(1+x[1,:]/cst.gamma))) + x[2,:]
    return xp

# system without control
def sys_nc(x):
    xp = torch.zeros(2,1)
    xp[0,:] = x[0,:] + cst.del_t*(-x[0,:]              + cst.Da1*      (1-x[0,:])*torch.exp(x[1,:]/(1+x[1,:]/cst.gamma)) + (1-cst.zeta)*x[0,:])
    xp[1,:] = x[1,:] + cst.del_t*(-x[1,:]*(1+cst.beta) + cst.Da2*cst.B*(1-x[0,:])*torch.exp(x[1,:]/(1+x[1,:]/cst.gamma)))
    return xp

x_u_xp = torch.zeros(5,cst.u_step*cst.x_step*cst.x_step)
x_u_xp[0,:] = torch.linspace(cst.x1_min,cst.x1_max,cst.x_step).kron(torch.ones(1,cst.x_step)).kron(torch.ones(1,cst.u_step))
x_u_xp[1,:] = torch.ones(1,cst.x_step).kron(torch.linspace(cst.x2_min,cst.x2_max,cst.x_step)).kron(torch.ones(1,cst.u_step))
x_u_xp[2,:] = torch.ones(1,cst.x_step*cst.x_step).kron(torch.linspace(cst.u_min,cst.u_max,cst.u_step))
x_u_xp[3:5,:] = sys(x_u_xp)
print("1st step done")
j = torch.zeros(cst.u_step*cst.x_step*cst.x_step,2,2)
for i in range(cst.u_step*cst.x_step*cst.x_step):
    temp = torch.autograd.functional.jacobian(sys_nc,x_u_xp[0:2,[i]])
    j[i,0,0] = temp[0][0,0]
    j[i,0,1] = temp[0][0,1]
    j[i,1,0] = temp[1][0,0]
    j[i,1,1] = temp[1][0,1]
print("2st step done")

torch.save(x_u_xp,'./data/x_u_xp.pt')
torch.save(j,'./data/j.pt')

print()