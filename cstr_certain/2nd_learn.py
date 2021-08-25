import torch
import cst
import time
def myloss(x_u_xp,A,B,epsilon):

    step_1_nn_out = rie_k(x_u_xp[:,0:2])
    step_2_nn_out = rie_k(x_u_xp[:,3:5])
    M_k  = torch.zeros(cst.u_step*cst.x_step*cst.x_step,2,2).cuda()
    M_kp = torch.zeros(cst.u_step*cst.x_step*cst.x_step,2,2).cuda()
    K_k  = torch.zeros(cst.u_step*cst.x_step*cst.x_step,1,2).cuda()
    M_k[:,0,0] = step_1_nn_out[:,0]
    M_k[:,0,1] = step_1_nn_out[:,1]
    M_k[:,1,0] = step_1_nn_out[:,1]
    M_k[:,1,1] = step_1_nn_out[:,2]
    K_k[:,0,0] = step_1_nn_out[:,3]
    K_k[:,0,1] = step_1_nn_out[:,4]
    M_kp[:,0,0] = step_2_nn_out[:,0]
    M_kp[:,0,1] = step_2_nn_out[:,1]
    M_kp[:,1,0] = step_2_nn_out[:,1]
    M_kp[:,1,1] = step_2_nn_out[:,2]

    ctr_con = - (A + B.matmul(K_k)).transpose(1,2).matmul(M_kp).matmul(A + B.matmul(K_k)) + 0.99*M_k

    con1 = -(M_k[:,0,0] - epsilon)
    con2 = -(M_k.det() - epsilon)
    con3 = -(ctr_con[:,0,0] - epsilon)
    con4 = -(ctr_con.det() - epsilon)
    loss = torch.max(torch.zeros(cst.u_step*cst.x_step*cst.x_step).cuda(),con1).cuda() + torch.max(torch.zeros(cst.u_step*cst.x_step*cst.x_step).cuda(),con2).cuda() + torch.max(torch.zeros(cst.u_step*cst.x_step*cst.x_step).cuda(),con3).cuda() + torch.max(torch.zeros(cst.u_step*cst.x_step*cst.x_step).cuda(),con4).cuda()
    loss = sum(loss)
    return loss

x_u_xp = torch.load("./data/x_u_xp.pt").cuda()
A = torch.load("./data/j.pt").cuda()
B = torch.tensor([[0.],[1.]]).cuda()
# make sure all things in gpu
x_u_xp = x_u_xp.T
A = A
epsilon = 0.01

rie_k = torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.ReLU(),
    torch.nn.ReLU(),
    torch.nn.Linear(10,5)
).cuda()

optimizer = torch.optim.AdamW(rie_k.parameters(), lr=0.05, betas=(0.1, 0.9), eps=1e-1, weight_decay=0.01, amsgrad=False)
# torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1, verbose=False)
time_start = time.time()
for i in range(10000):
    loss = myloss(x_u_xp,A,B,epsilon)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss)
    if loss <= 0.9*epsilon:
        break
time_end = time.time()
print(time_end-time_start)
torch.save(rie_k,'./data/rie_k.pkl')
print()
