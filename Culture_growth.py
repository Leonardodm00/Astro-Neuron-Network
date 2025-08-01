import os
directory = r"C:\Users\leona\Desktop\PhD\PYTHON\Neuronal culture"  # Replace with your desired path
os.chdir(directory)

import time
import pandas as pd
import numpy as np

from Culture_class import *

# AGGRAGATES
cov_val_mean = 0.06
std_cov_val = 0.02
n_cluster = int(5)

#FRACTALS
cov1_fractals=0.02
cov2_fractals=0.005
# for r in RHO:


# GENERAL 
x_size = 1
y_size = 1
z_size = 0.3
rho = 8
delta = 0.05
R_influence = 0.3
over_factor = 100
Dl = 1e-3
#%
bias_GABA = 400e-3
bias_GLU = 1200e-3
Number_GABA = 4
Number_GLU = 1

# bias for synapse connections: py-py, py-inh, inh-inh/py
alpha_values =  [0.8,1.2,1] 
Gamma_GABA = 1
Gamma_GLU = 1

# Suitable value 
P_b = 0.007
Branch_TH = 0.01 # [mm] New branches of length lower that TH are discarded
# Maybe in the future I can implement cell-specific P_b
# P_b_GLU = 0.002 #branching probability
# P_b_GABA = 0.01

Type = 1   # 1= HOMOGENEOUS,2 = AGGREGATES, 3 = FRACTALS
excitatory_persentage = 100/100
N= int(x_size*y_size*z_size*rho)



network = Network_3D(rho, 7.5e-3, x_size, y_size,
                     z_size, excitatory_persentage,delta)
X, Y, Z = network.place_neurons(
    Type, n_cluster, cov_val_mean, std_cov_val, cov1_fractals,cov2_fractals,cent=[],predet=1)



Somata = np.zeros((N,4))
result_vstack = np.vstack((X, Y, Z)).T # Transpose to get Nx3
num =np.arange(N)
Somata[:,:3] = result_vstack
Somata[:,3] = num


vect_field,Vector_field_surrogate = network.generate_vector_field(Somata,R_influence,over_factor)

#%%

start_time = time.time()  # Record the start time


A,Xi,Yi,Zi,Conn = network.grow_W(Number_GABA,Number_GLU,bias_GABA,bias_GLU,alpha_values,Gamma_GABA,Gamma_GLU,P_b,Dl,Branch_TH)


end_time = time.time()  # Record the end time

elapsed_time = end_time - start_time  # Calculate the elapsed time
print(f"Elapsed time: {elapsed_time:.4f} seconds")  # Format the output
#%%

%matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


x_arrays, y_arrays, z_arrays = extract_nonzero_arrays_separately(Xi, Yi, Zi)
N = len(x_arrays)  # Number of segments
Ne = N*excitatory_persentage
for i in range(N):
    
    x1 = x_arrays[i]
    y1 = y_arrays[i]
    z1 = z_arrays[i]
    
    
    for k in range(len(x1) - 1):
        
            ax.plot([x1[k], x1[k+1]], [y1[k], y1[k+1]], [z1[k], z1[k+1]], color='red', label='Segments')
    
r_dendrite_mu   = 120e-3 
# Draw the spheres
# for i in range(len(X)):
#     # draw sphere
#     u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#     x = r_dendrite_mu* np.outer(np.cos(u),np.sin(v))  + X[i]
#     y =r_dendrite_mu* np.outer(np.sin(u),np.sin(v))  + Y[i]
#     z =r_dendrite_mu* np.outer(np.ones(np.size(u)),np.cos(v))  + Z[i]
#     ax.plot_surface(x, y, z, color='r',alpha = 0.1) 
# COL = ['g','orange', 'purple', 'brown', 'pink']
# for i in range(len(X)):
#     # draw sphere
#     u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#     x = R_influence* np.outer(np.cos(u),np.sin(v))  + X[i]
#     y =R_influence* np.outer(np.sin(u),np.sin(v))  + Y[i]
#     z =R_influence* np.outer(np.ones(np.size(u)),np.cos(v))  + Z[i]
#     ax.plot_surface( x, y, z,  rstride=1, cstride=1, color='c', alpha=0, linewidth=0)    
ax.scatter(X,Y,Z,s=10)
# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title(f'Branching probability {P_b}')
ax = plt.gca()  # Get the current axes object
ax.set_xlim([-0.5, 1.5])
ax.set_ylim([-0.5, 1.5])
ax.set_zlim([-0.5, 1.5])

# Show the plot
plt.show()
