import numpy as np
import torch 
import torch.nn as nn
import scipy 
import scipy.optimize as opt
import matplotlib
import cst
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rc

# model learnt from data stored in rie_k
rie_k = torch.load('./data/rie_k.pkl').cpu()
def intgral_delta_u(r,d,N,u0):
    u = u0
    x = r
    for n in range(N):
        delta_x = d[:,[n]]
        # calculate the output
        nn_output = rie_k(torch.tensor(x).t().float())
        # change output 2d tensor to 2d numpy
        K = nn_output.detach().numpy()
        u = new_u(u, delta_x, K)
    return u

def new_u(u, delta_x, K):
    K = K[0,3:5]
    u = u + np.matmul(K,delta_x)
    return u

# r numpy 2*1
# xt current x numpy 2*1
# N how refine are steps
# return d 1*2N
def geodesic(r,xt,N):
    d0 = np.append((xt[0,0] - r[0,0])/N*np.ones(N),(xt[1,0] - r[1,0])/N*np.ones(N))
    A = np.ones([N,N]) 
    A = np.tril(A,-1) + np.diag(np.diag(A))
    cons = [{'type': 'eq', 'fun': lambda d: sum(d[0:N])   - xt[0,0] + r[0,0]},
            {'type': 'eq', 'fun': lambda d: sum(d[N:2*N]) - xt[1,0] + r[1,0]}]

    for i in range(0, N):
        cons.append({'type': 'ineq', 'fun': lambda d: ( A[i, ].dot(d[0:N])   - cst.x1_max)})
        cons.append({'type': 'ineq', 'fun': lambda d: ( A[i, ].dot(d[N:2*N]) - cst.x2_max)})
        cons.append({'type': 'ineq', 'fun': lambda d: (-A[i, ].dot(d[0:N])   + cst.x1_min)})
        cons.append({'type': 'ineq', 'fun': lambda d: (-A[i, ].dot(d[N:2*N]) + cst.x2_min)})
        cons.append({'type': 'ineq', 'fun': lambda d: ( np.eye(2*N).dot(d)   - 2/N)})
        cons.append({'type': 'ineq', 'fun': lambda d: (-np.eye(2*N).dot(d)   + 2/N)})

    cons = tuple(cons)
    d = opt.minimize(lambda d:rie_path(r,d,N),d0,constraints=cons)
    return np.array([d.x[0:N],d.x[N:2*N]]), d.fun

# path integral
# input reference r numpy 2*1
# d is determin varible in 1*2N
# N is how refine is steps
def rie_path(r,d,N):
    path_cost = 0
    # change d to delta_x which is 2*N in dim
    delta_x = np.array([d[0:N],d[N:2*N]])
    # define how small a step is
    delta_s = 1/N
    # initial point is x = r
    x = r
    # integral along x
    for n in range(N):
        path_cost = torch.zeros([1,1])
        # Riemannian distance in small step, self[:,[n]] 
        # to remain dim
        quad_cost = rie_metric(delta_x[:,[n]],delta_s,x)
        # accumulate using Riemannian approximation
        path_cost = path_cost + np.sqrt(quad_cost)
        # update x
        x = x + delta_x[:,[n]]
    return path_cost[0][0]

# energy integral
# input r numpy 2*1
# d is delta_x
def rie_energy(r,d,N):
    x = r
    # define how small a step is
    delta_s = 1/N
    ene = 0
    for n in range(N):
        quad_cost = rie_metric(d[:,[n]],delta_s,x)
        ene = ene + delta_s * quad_cost
        # update x
        x = x + d[:,[n]]
    return ene
# delta_x is np collomn vector
# delta_s is a scallar
# x is a numpy collomn vector

def rie_metric(delta_x,delta_s,x):
    partial_x_partial_s = delta_x/delta_s
    # neural requairs raw tensor as input
    nn_output = rie_k(torch.tensor(x).t().float())
    # change output 2d tensor to 2d numpy
    M = nn_output.detach().numpy()
    # select first three elements to form the Metirc M
    M = np.array([[M[0,0],M[0,1]],[M[0,1],M[0,2]]])
    # return delta_x'*M*delta_x
    output = np.matmul(partial_x_partial_s.T,np.matmul(M,partial_x_partial_s))*delta_s
    # if output <0:
    #     output = 0
    return output 

# system dynamic 
# x is collomn np vector
# u is a np scallar
def sys_d_c(x,u):
    xp = torch.zeros([2,1])
    xp[0,:] = x[0,:] + cst.del_t*(-x[0,:]              + cst.Da1*      (1-x[0,:])*torch.exp(x[1,:]/(1+x[1,:]/cst.gamma))  + (1-cst.zeta)*x[0,:])
    xp[1,:] = x[1,:] + cst.del_t*(-x[1,:]*(1+cst.beta) + cst.Da2*cst.B*(1-x[0,:])*torch.exp(x[1,:]/(1+x[1,:]/cst.gamma))) + u
    return xp

def sys_nn(x,u,b,T_MAX):
    xp = torch.zeros([2,T_MAX-1])
    xp[0,:] = x[0,:] + cst.del_t*(-x[0,:]              + cst.Da1*  (1-x[0,:])*torch.exp(x[1,:]/(1+x[1,:]/cst.gamma))  + (1-cst.zeta)*x[0,:])
    xp[1,:] = x[1,:] + cst.del_t*(-x[1,:]*(1+cst.beta) + cst.Da2*b*(1-x[0,:])*torch.exp(x[1,:]/(1+x[1,:]/cst.gamma))) + u
    return xp

# Construct a NN to 
def myloss(nn_out,x_current,u,x_next,K):
    x_nn_next = sys_nn(x_current,u,nn_out,K)
    loss = x_nn_next - x_next
    loss = loss[0,:]**2 + loss[1,:]**2
    loss = torch.sum(loss)
    return loss

b_nn = torch.nn.Sequential(
    torch.nn.Linear(4,2),
    torch.nn.ReLU(),
    torch.nn.Linear(2,1)
)

print()
T_MAX = 50
N = 10
r  = torch.zeros([2,T_MAX])
x  = torch.zeros((2,T_MAX))
u  = torch.zeros((1,T_MAX))
u0 = torch.zeros([1,T_MAX])
x[:,0] = torch.tensor([0.8,0.8]) 

ene     = torch.zeros((1,T_MAX))
ene_bnd = torch.zeros((1,T_MAX))
R_B = torch.zeros((1,T_MAX))
E_B = torch.ones((1,T_MAX))
ERROR_B = torch.ones((1,T_MAX))
x_nn = torch.zeros((2,T_MAX))
optimizer = torch.optim.AdamW(b_nn.parameters(), lr=0.002, betas=(0.1, 0.9), eps=1e-2, weight_decay=0.01, amsgrad=False)

r[:,0] = torch.tensor([0.5706,0.5706])
cst.B = 1
for t in range(0,T_MAX-1): 
    u0[0,[t]]   = 0 
    r[: ,[t+1]] = sys_d_c(r[:,[t]],u0[0,[t]])
cst.B = 1.13
# simulation
for t in tqdm(range(T_MAX-1)):
    R_B[0,t] = cst.B
    xt = x[:,[t]]
    rt = r[:,[t]]
    d, path = geodesic(rt,xt,N)
    ene[0,t] = pow(path,2)
    ene_bnd[0,t] = ene[0,0]*(0.99**(t))
    u[0,t] = intgral_delta_u(rt,d,N,u0[0,[t]])    
    x[:,[t+1]] = sys_d_c(x[:,[t]],u[0,t])
    if t > 20:
        K = 3
        data = torch.cat((x[0,t-K:t-1],x[1,t-K:t-1]),0)
        data_x = torch.stack((x[0,t-K:t-1],x[1,t-K:t-1]),0)
        data_xp= torch.stack((x[0,t-K+1:t],x[1,t-K+1:t]),0)
        data_u = u[0,t-K:t-1]
        for i in range(3000):
            nn_out = b_nn(data)
            loss = myloss(nn_out,data_x,data_u,data_xp,K)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if loss <= 1e-10:
                break
        E_B[:,t+1] = nn_out.detach()
        r[: ,[t+1]] = sys_nn(r[:,[t]],u0[0,[t]],nn_out.detach(),2)
cst.B = 1.1 + 0.1*torch.sin(0.01*torch.tensor(T_MAX))
R_B[0,T_MAX-1] = cst.B    
for t in tqdm(range(T_MAX)):
    xt = x[:,[t]]
    rt = r[:,[t]]
    d, path = geodesic(rt,xt,N)
    ene[0,t] = pow(path,2)
    ene_bnd[0,t] = ene[0,0]*(0.99**(t))

# # T_MAX = 11
# # T_MAX = 11
# data = torch.cat((x[0,0:10],x[1,0:10]),0)
# data_x = torch.stack((x[0,0:10],x[1,0:10]),0)
# data_xp= torch.stack((x[0,1:11],x[1,1:11]),0)
# data_u = u[0,0:10]
# for i in range(5000):
#     nn_out = b_nn(data)
#     loss = myloss(nn_out,data_x,data_u,data_xp)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     if loss <= 1e-3:
#         break
t = np.linspace(0,1,T_MAX)
plt.figure(1)
plt.plot(t,r[0,:],'r',linewidth=1.15,linestyle= "--",label = '$x^*_1$')
plt.plot(t,x[0,:],'r',linewidth=1.15,label = '$x_1$')
plt.plot(t,r[1,:],'b',linewidth=1.15,linestyle= "--",label = '$x^*_2$')
plt.plot(t,x[1,:],'b',linewidth=1.15,label = '$x_2$')
plt.plot(t,u[0,:],'g',linewidth=1.15,label = '$u$')
plt.legend()
plt.xlabel('Time')
plt.ylabel('State(Effort) Value')
plt.savefig("pics_para_sim/NN_C_Responds.pdf")

plt.figure(2)
plt.step(t,ene[0,:],'r',label = 'Riemannian Energy')
plt.step(t,ene_bnd[0,:],'b',label = 'Riemannian Energy Boundary')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Energy Value')
plt.savefig("pics_para_sim/NN_C_R.pdf")

plt.figure(3)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"})
plt.subplots(nrows=4, ncols=1, sharex=True)
plt.subplot(4, 1, 1)
plt.step(t,r[0,:],'b',linewidth=1.15,linestyle= 'dotted',label = '$x^*_1$')
plt.step(t,x[0,:],'r',linewidth=1.15,label = '$x_1$')
plt.xticks([])
plt.legend(loc = 'right')
plt.subplot(4, 1, 2)
plt.step(t,r[1,:],'b',linewidth=1.15,linestyle= 'dotted',label = '$x^*_2$')
plt.step(t,x[1,:],'r',linewidth=1.15,label = '$x_2$')
plt.xticks([])
# plt.ylabel('State(Effort) Value')
plt.legend(loc = 'right')
plt.subplot(4, 1, 3)
plt.step(t,u[0,:],'g',linewidth=1.15,label = '$u$')
plt.xticks([])
plt.legend()
plt.subplot(4, 1, 4)
t = np.linspace(0,1,T_MAX-1)
ERROR_B = R_B - E_B
plt.step(t,ERROR_B[0,0:T_MAX-1],'r',linewidth=1.15,label = '$r-\hat{r}$')
plt.legend()
plt.xlabel('$Time (h)$')
plt.savefig("pics_para_sim/NN_C_Responds_sub.pdf")

plt.figure(4)
plt.step(t,ERROR_B[0,0:T_MAX-1],'r',linewidth=1.15,label = '$x^*_1$')
# plt.step(t,R_B[0,:],'r',label = 'Real Value')
# plt.step(t,E_B[0,:],'b',label = 'Estimation')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Energy Value')
plt.savefig("pics_para_sim/Estimation.pdf")
print()