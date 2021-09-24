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



print()
T_MAX = 100
T_1 = 50 
N = 10
r  = torch.zeros([2,T_MAX])
x  = torch.zeros((2,T_MAX))
u  = torch.zeros((1,T_MAX-1))
u0 = torch.zeros([1,T_MAX-1])
x[:,0] = torch.tensor([0.6,0.6])

ene     = torch.zeros((1,T_MAX))
ene_bnd = torch.zeros((1,T_MAX))
R_B = torch.zeros((1,T_MAX))
E_B = torch.ones((1,T_MAX))
ERROR_B = torch.ones((1,T_MAX))

cst.B = 1
r[:,0] = torch.tensor([0.9394,0.2970])
for t in range(0,T_1): 
    u0[0,[t]]   = 0.05
    r[: ,[t+1]] = sys_d_c(r[:,[t]],u0[0,[t]])
r[:,T_1] = torch.tensor([0.9454,0.5473])
for t in range(T_1,T_MAX-1): 
    u0[0,[t]]   = 0.1
    r[: ,[t+1]] = sys_d_c(r[:,[t]],u0[0,[t]])
# cst.B = 1.2
# simulation
for t in tqdm(range(T_MAX-1)):
    xt = x[:,[t]]
    rt = r[:,[t]]
    d, path = geodesic(rt,xt,N)
    ene[0,t] = pow(path,2)
    ene_bnd[0,t] = ene[0,0]*(0.99**(t))
    u[0,t] = intgral_delta_u(rt,d,N,u0[0,[t]])    
    x[:,[t+1]] = sys_d_c(x[:,[t]],u[0,t])
for t in tqdm(range(T_MAX)):
    xt = x[:,[t]]
    rt = r[:,[t]]
    d, path = geodesic(rt,xt,N)
    ene[0,t] = pow(path,2)
    ene_bnd[0,t] = ene[0,0]*(0.99**(t))
for t in tqdm(range(T_1,T_MAX)):
    ene_bnd[0,t] = ene[0,T_1]*(0.99**(t-T_1))

t = np.linspace(0,1,T_MAX)

# plt.figure(3)
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif"})
# plt.plot(r[0,:],r[1,:],'.--',color='b',label = '$x^*$')
# plt.plot(x[0,:],x[1,:],'.-',color='r',label = '$x$')
# plt.legend()
# plt.savefig("pics_sim/NN_C_Phase.pdf")

plt.figure(4)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"})
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
params= {'text.latex.preamble' : [r'\usepackage{yhmath}']}
plt.rcParams.update(params)

plt.subplots(nrows=3, ncols=1, sharex=True)
plt.subplot(2, 1, 1)
plt.step(t,r[0,:],'r',linewidth=1.15,linestyle= 'dotted',label = '$x^*_1$')
plt.step(t,x[0,:],'r',linewidth=1.15,label = '$x_1$')
plt.step(t,r[1,:],'b',linewidth=1.15,linestyle= 'dotted',label = '$x^*_2$')
plt.step(t,x[1,:],'b',linewidth=1.15,label = '$x_2$')
plt.step(t[0:T_MAX-1],u[0,:],'g',linewidth=1.15,label = '$u$')
plt.legend(loc = 'right')
plt.xticks([])
plt.subplot(2, 1, 2)
plt.step(t,ene[0,:],'r',label = '$d(\gamma)$')
plt.step(t,ene_bnd[0,:],'b',label = '$\wideparen{d(\gamma)}$')
# plt.legend(loc = 'right')
plt.legend()

# plt.xticks([])
plt.xlabel('$Time (h)$')
plt.savefig("pics_sim/CSTR_SIM.pdf")
print()