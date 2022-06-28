## Discrete-time Contraction-based Control of Nonlinear Systems with Parametric Uncertainties using Neural Networks

This is the simulation for arxiv artilc   (https://arxiv.org/abs/2105.05432).

The simulation requires 
- PyTorch 1.90+;
- Python 3.7+;
- Packages in the code;
- Opening one of the 4 simulation as workspace.

### Some explaination of the code
In 1st_data_gen.py, you will see some code like follows,
```python
x_u_b_xp = torch.zeros(6,cst.u_step*cst.x_step*cst.x_step*cst.b_step)
x_u_b_xp[0,:] = torch.linspace(cst.x1_min,cst.x1_max,cst.x_step).kron(torch.ones(1,cst.x_step)).kron(torch.ones(1,cst.u_step)).kron(torch.ones(1,cst.b_step))
x_u_b_xp[1,:] = torch.ones(1,cst.x_step).kron(torch.linspace(cst.x2_min,cst.x2_max,cst.x_step)).kron(torch.ones(1,cst.u_step)).kron(torch.ones(1,cst.b_step))
x_u_b_xp[2,:] = torch.ones(1,cst.x_step*cst.x_step).kron(torch.linspace(cst.u_min,cst.u_max,cst.u_step)).kron(torch.ones(1,cst.b_step))
x_u_b_xp[3,:] = torch.ones(1,cst.x_step*cst.x_step).kron(torch.ones(1,cst.u_step)).kron(torch.linspace(cst.b_min,cst.b_max,cst.b_step))
```
this use the Kronecker product to generate combinations of possible x,u and r. 
