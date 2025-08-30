
import os
directory = r"C:\Users\Admin\Desktop\Leonardo\ASN\Growth"  # Replace with your desired path
os.chdir(directory)

import time
import pandas as pd
import numpy as np

from Culture_class import *
if __name__ == "__main__": 
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
    z_size = 1
    rho = 40
    delta = 0.05
    R_influence = 0.3
    over_factor = 100
    Dl = 1e-2
    #%
    bias_GABA = 500e-3
    bias_GLU = 600e-3
    Number_GABA = 4
    Number_GLU = 1
    
    # bias for synapse connections: py-py, py-inh, inh-inh/py
    alpha_values =  [0.8,1.2,1] 
    Gamma_GABA = 1
    Gamma_GLU = 1
    
    # Suitable value 
    P_b = 0.0
    Branch_TH = 0.01 # [mm] New branches of length lower that TH are discarded
    # Maybe in the future I can implement cell-specific P_b
    # P_b_GLU = 0.002 #branching probability
    # P_b_GABA = 0.01
    
    Type = 1   # 1= HOMOGENEOUS,2 = AGGREGATES, 3 = FRACTALS
    excitatory_persentage = 30/100
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


# vect_field,Vector_field_surrogate = network.generate_vector_field(Somata,R_influence,over_factor)

#%%




    start_time = time.time()  # Record the start time
    
    
    out_exc,out_inh,Excitatory_cells_idx,Inhibitory_cells_idx = network.grow_W(Number_GABA,Number_GLU,bias_GABA,bias_GLU,alpha_values,Gamma_GABA,Gamma_GLU,P_b,Dl,Branch_TH,)
    
    
    end_time = time.time()  # Record the end time
    
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")  # Format the output
#%%
# -------------------------- POST PROCESSING --------------------------


    Neurites_x = []
    Neurites_y = []
    Neurites_z = []
    for j in range(len(out_exc)):
        
        Neurites_x.append(out_inh[j][1])
        Neurites_y.append(out_inh[j][2])
        Neurites_z.append(out_inh[j][3])
        
        
        # Neurites_x.append(out_exc[j][1])
        # Neurites_y.append(out_exc[j][2])
        # Neurites_z.append(out_exc[j][3])
    
    
    Neurites_x = np.vstack(Neurites_x)
    Neurites_y = np.vstack(Neurites_y)
    Neurites_z = np.vstack(Neurites_z)
    
    
    
    
    # %matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    def pad_vectors_to_max_length(v1, v2, v3):
        """
        Pads three input vectors with their last value to match the length of the longest vector.
    
        Args:
            v1 (np.ndarray): The first input vector.
            v2 (np.ndarray): The second input vector.
            v3 (np.ndarray): The third input vector.
    
        Returns:
            tuple: A tuple containing the three padded vectors (padded_v1, padded_v2, padded_v3).
        """
        vectors = [v1, v2, v3]
        max_len = max(len(v) for v in vectors)
        
        padded_vectors = []
        for v in vectors:
            current_len = len(v)
            if current_len < max_len:
                # Determine the value to pad with (the last element)
                last_value = v[-1]
                # Calculate how many values need to be added
                num_to_pad = max_len - current_len
                # Create the padding array
                padding = np.full(num_to_pad, last_value)
                # Concatenate the original vector with the padding
                padded_v = np.concatenate((v, padding))
                padded_vectors.append(padded_v)
            else:
                padded_vectors.append(v)
                
        return tuple(padded_vectors)
    x_arrays, y_arrays, z_arrays = extract_nonzero_arrays_separately(Neurites_x, Neurites_y, Neurites_z)
    N = len(x_arrays)  # Number of segments
    Ne = N*excitatory_persentage
    for i in range(len(x_arrays)):
        
        x1 = x_arrays[i]
        y1 = y_arrays[i]
        z1 = z_arrays[i]
        
        x1,y1,z1 = pad_vectors_to_max_length(x1, y1, z1)
        
        
        
        for k in range(len(x1) - 1):
            
                ax.plot([x1[k], x1[k+1]], [y1[k], y1[k+1]], [z1[k], z1[k+1]], color='red', label='Segments', linewidth=2)
        
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
    ax.scatter(X[0:N],Y[0:N],Z[0:N],s=10)
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







