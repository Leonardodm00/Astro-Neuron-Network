
import random
import math
from tqdm import tqdm
import imageio
from pathlib import Path  # Recommended for path handling
import csv
import os
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.neighbors import BallTree
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree


directory = r"C:\Users\Admin\Desktop\Leonardo\ASN\Growth"  # Replace with your desired path
os.chdir(directory)
from Functionalities import *




class Network_3D:

     def __init__(self, rho, r_soma, width, depth, height, excitatory_persentage,delta):

         self.rho = rho

         self.r_soma = r_soma
         self.delta = delta
         self.width = width     # culture width (mm)
         self.height = height    # culture height (mm)
         self.depth = depth     # culture depth (mm)
         self.excitatory_persentage = excitatory_persentage

         # NO OBSTACLES
     # Method istances
     
     def generate_dendritic_arbour(max_rad,dx,interval):
         '''
         

        Parameters
        ----------
        max_rad : integer. 220[um]
            Maximum radius from the soma to calculate the probability of presence of 
            a dendritic process
        dx : integer. 0.0001
            Spacing between sample points for the trapezoidal method

        interval : integer. 5
            Step in computing the shell ranges (r_0 and r_1) from 0 to max_rad.
            Must be an integer of thte latter

        Returns
        -------
        Syn_prob : array (N,)
            Array of probability of connections for a axonal neurite 
        
         rarius_val : array (N,)
             Distances from the soma where the relative elements in 'Syn_prob'
             have been calculated
        
         Where N is max_rad/interval
         '''
         
     
         
         
         max_rad = 220
         rarius_val = np.arange(0,max_rad,interval)
         Syn_prob = np.zeros(len(rarius_val))
         k= 0
         dx = 0.0001
         for i in np.arange(1,len(rarius_val)):
             
           r_1 = rarius_val[i]
           r_0 = rarius_val[i-1]
           Syn_prob[k] = get_dendrite_prob(r_1,r_0,dx)
           k = k+1
           
         plt.figure()
         plt.title('Synapse probability')
         plt.plot(rarius_val,Syn_prob)
         plt.xlabel('Distance from soma [um]')
         plt.ylabel('Probability')
         # plt.yscale('log')
         plt.xlim(0,200)
         
         
         
         return Syn_prob,rarius_val
     
     def generate_vector_field(self,Somata,R_influence,over_factor):
         
         ############################################################
         
         # R_influence :Influence radius to define the vector field [mm]
         # Somata: [X,Y,Z,Type]coordinates in[mm] (Need to be correctly mapped in the field) and 
         #                     type is the neurons type
         #         the mapped values are the original with n-th number of decimals, 2 if
         #         the space has been oversampled by a factor of 100. Np.Array [N,3].
         # over_factor: over-sampling factor.
           
         # REMARK: the culture sizes are given in mm, this mean that setting the over_factor equal
         #         to 1000 we are defining a space with micrometer accuracy (also can be thought of as 
         #         a discretizationa t the micrometer space).
         ############################################################
         
         ############    REMARKS     ############
         # Multiple influence clouds acting on the same node. The growing axon will have to choose between 
         # multiple gradients in base of the following gerarchica conditions: 1) Type of neuron 2) Magnitude of the gradient
         
         delta = self.delta
         width = self.width     # culture width (mm)
         height = self.height    # culture height (mm)
         depth = self.depth
         self.over_factor = over_factor
         
         
         # Scale the Somata values in order ot be mapped into the vector field space
         Somata_scaled = Somata * over_factor
         Somata_scaled =np.array(Somata_scaled,dtype=int)
         
         
         N_neu = Somata.shape[0]
         
         
         ####### CHANGE ##############
         # Set the vector field. In addition generate an external vector field of thickness 'delta' to impede the 
         # processes to surpass the culture's extremities (done in ### const ###)
         
         X_over = np.arange(0,int((width)*over_factor),1,dtype=int)
         Y_over = np.arange(0,int((depth)*over_factor),1,dtype=int)
         Z_over = np.arange(0,int((height)*over_factor),1,dtype=int)
         
         
         # I generate the variable that will contain the vectors that point toward the dendritic arbour center (as
         # if we are modelling a basin of attraction)
         
         dimensions = (X_over.size, Y_over.size, Z_over.size)  # (depth, rows, cols)
         
         # Columns:0-2 is the three-dimensional vector ; 3 is the type of neuron at the center of the basin of attraction
         # Is used to embed preferances among neuron sub-types.
         Vector_field = np.zeros(dimensions,dtype = list)
         
         # I generate the surrogate of the vector field in which are contained all the points' positions
         # along 0,1,2 columns
         
         
         
         Vector_field_surrogate = np.zeros(((X_over.size*Y_over.size*Z_over.size),3))
         Num_a = X_over.size*Y_over.size*Z_over.size
         p = 0 # keep traks of the points
         for x in X_over:
             for y in Y_over:
                 for z in Z_over:
                     Vector_field_surrogate[p,0]=x
                     Vector_field_surrogate[p,1]=y
                     Vector_field_surrogate[p,2]=z
                     p = p+1
                     

        
        ############## DEPRECATED ##############

        # ### I need 1 vector field containing the whole culture plus the boundaries
        # #   An array containing all the countur points and an array containing all the 
        # #   points at the boundaries
        
        # # TO construct the boundary vector filed for each point I'll take the one in the culture
        # # that is directly below/above it inorder to have conturn vectors all parrallel to each
        # # other across the same culture's face
         
        #  # Counturn points off the culutre
         
         
        #  delta1 = delta
        #  Countur_points,N_points_x,N_points_y,N_points_z = Get_cube_faces(0,width,depth,height,over_factor,delta1,X_over,Y_over,Z_over)
         
        #  #  most external points still belonging to the culture
         
        #  # delta1 = delta
        #  # External_points = Get_cube_faces(delta,width-delta,depth-delta,height-delta,over_factor,delta1,X_over,Y_over,Z_over)

         
       
        #  # Use cKDTree for efficient nearest neighbor search
         
        #  # External_points_tree = cKDTree(External_points)
         
         
        #  # The position points must be scaled with regard to Dl to obtain the same measure scale than 
        #  # the points
        #  # Vector_field_surrogate = cKDTree(Vector_field_surrogate)
         
         
        #  Init_vec = np.array((0,N_points_x,N_points_y))
        #  End_vec  = np.array((N_points_x,N_points_y,N_points_z))
         
        #  k = 0
        #  for N_points_ in [N_points_x,N_points_y,N_points_z]:
             
           
           
        #     Init = Init_vec[k]
        #     End = End_vec[k]
           
        #     Vector_field = Load_countur_vectors(Vector_field,Countur_points,Init,End)
           
        #     k = k+1

        #####################################################
         
         
             
             
         

         
         Eps = 1e-6
         for neu in range(N_neu):
             # print('A')
             # Generate a mask for the points within the influence region
             Mask = np.sqrt(np.power(Vector_field_surrogate[:,0]-Somata_scaled[neu,0], 2)+np.power(Vector_field_surrogate[:,1]-Somata_scaled[neu,1], 2)+np.power(Vector_field_surrogate[:,2]-Somata_scaled[neu,2], 2)) < R_influence*over_factor
             
             # Find the points within it
             Indecies =  np.where(Mask)[0]
             # print('Num of points:', Indecies.size)
             # For each point I find the vector pointing towards the dendrite center.
                   
           
             for Idx in Indecies:
            
                 # Draw point's coordinate
                 Point = [Vector_field_surrogate[Idx,0] ,Vector_field_surrogate[Idx,1], Vector_field_surrogate[Idx,2]]
                 
                 # Immediatly check whether the point is a contour point
                 
                 VF_array = Vector_field[int(Vector_field_surrogate[Idx,0]) ,int(Vector_field_surrogate[Idx,1]), int(Vector_field_surrogate[Idx,2])]
                 
                 
                 # if ~np.all(VF_array == 0) and VF_array[0,3] == -1: # The point is the boundary region
                 
                 #     continue
                 

                 
                 
                 # If the somata coincides with a point in the vector field space the relative vector is wrongly 
                 # normalized per zero --> nan elements
                 # Eps is the tollerance that we admint
                 
                 if np.linalg.norm(Somata_scaled[neu,0:3] - Point) < Eps:    # Calculate Euclidean distance
                     
                     continue
                 
              
                 # Define the vector
                 Vector = Somata_scaled[neu,0:3] - Point
                 # print('Somata_scaled[neu,0:3]',Somata_scaled[neu,0:3])
                 # print('Point',Point)
                 # print('Vector',Vector)
                 # print('                     ')
                 # print('                     ')
                 Vector_norm = Vector/np.linalg.norm(Vector)
                 
                 # print('np.linalg.norm(Vector)',np.linalg.norm(Vector))
                 # print('Vector_norm',Vector_norm)
                 
                 # print('                     ')
                 # print('                     ')
                 if np.isnan(Vector_norm).any():  # Check if ANY element is NaN
                     raise ValueError("NaN found in the array!")  # Or use sys.exit()
                 # The vector magnitude is taken from a truncated expontnatial function. More over we need to map 
                 # the distance value (which is in the vector_field scale) into the range of the exponential funciton.
                 distance = np.sqrt(np.power(Point[0]-Somata_scaled[neu,0], 2)+np.power(Point[1]-Somata_scaled[neu,1], 2)+np.power(Point[2]-Somata_scaled[neu,2], 2))
                 
                 # The range in which we map is between 0 and 1 wich results in lower value of approx 0.5 up to 1
                 mapped_value = map_range(distance, 0, R_influence*over_factor, 0, 1)
                 
                 
                 
                 # UNINDEND ######################
                 # Calculate the distance dependent magnitude of the vector
                 # Magnitude = np.exp(-mapped_value / 1)
                 Magnitude = 0
                 # Scale the vector
                 Vector_s = Vector_norm*Magnitude
                 
             
                 
                 ## Vector scale has 4 entries: first three are the vector elements, the 4th is the neuron of belonging 
                 Vector_scaled = np.zeros([1,5])
                 Vector_scaled[0,0:3] = Vector_s.reshape(1, 3)
                 Vector_scaled[0,3] = Somata[0,3]
                 Vector_scaled[0,4] = neu
                 
                 # Vector_scaled =  Vector_scaled.reshape(1, 3)
                 
                 # Handle multiple neurons' gradient acting on the same point
                 
                 
                 
                 if np.all(VF_array == 0):
                     
                 
                 # Load the vector in its relative point in the vector field
                     Vector_field[int(Vector_field_surrogate[Idx,0]) ,int(Vector_field_surrogate[Idx,1]), int(Vector_field_surrogate[Idx,2])] = Vector_scaled
                     # MMMM = MMMM+1
                 
                 
                 
                 else:
                     
                     result = np.concatenate((VF_array, Vector_scaled), axis=0)
                     Vector_field[int(Vector_field_surrogate[Idx,0]) ,int(Vector_field_surrogate[Idx,1]), int(Vector_field_surrogate[Idx,2])]=result
                     # nnnn = nnnn+1
          
         # print('First collocations:',MMMM) 
         # print('Second collocations:',nnnn)  
         self.Vector_Field = Vector_field
         self.Vector_field_surrogate = Vector_field_surrogate
         return   Vector_field, Vector_field_surrogate


     def place_neurons(self, Type, n_cluster, cov_mean, std_cov_val,cov1_fractals,cov2_fractals,cent,predet,  **kwargs):

         # Type = type of network: 1 = homogeneous, 2 = aggregates
         # cov_mean = mean value of element along the digaonal of the covariace matricies defined for the clusters
         # n_cluster = number of clusters
         # std_cov_val = standard deviation regarding the distribution from which the cov_values(alogn diagonal) are drawn
         # delta = nested volume in which neurons are generated, increase computational efficiency for futher growth
         # cov1_fractals,cov2_fractals= values insrted in the diagonal covariacen matrix that rules the spread of points
         #                               placed about the centers of grater and smaller cubes respectively.
         # Pram
         # Predet = 1 no centers present, 2 = center present
         width = self.width
         height = self.height
         depth = self.depth
         delta = self.delta
         rho = self.rho
         r_soma = self.r_soma

         for argn, argv in kwargs.items():
             match argn:
                 case 'rho':
                     rho = float(argv)
                 case 'r_soma':
                     r_soma = float(argv)

     # some derived parameters:
         M = int(height*width*depth*rho)   # number of neurons (3D)

         if Type == 2:
             delta = delta  # Better disporition of the centers, not generated attached to the border
             n_width = width-delta # 3 mm inside the effective volume
             n_depth = depth-delta
             n_height = height -delta

             # FOR AGGREAGTIONS
             
             if predet == 1:
             
             
                 # Define center and number of points
                 X_c, Y_c, Z_c = np.zeros((3, n_cluster))
                 X_c[0] = np.random.uniform(delta,n_width)
                 Y_c[0] = np.random.uniform(delta,n_depth)
                 Z_c[0] = np.random.uniform(delta,n_height)
                 for i in range(1, n_cluster):
         
                     X_c[i] = np.random.uniform(delta,n_width)
                     Y_c[i] = np.random.uniform(delta,n_depth)
                     Z_c[i] = np.random.uniform(delta,n_height)
         
                 centers = np.vstack((X_c, Y_c, Z_c))
                 centers = centers.T
             
             elif predet == 2:
                 centers = cent

             # In this way the clusters will have different compactness
             std_val = std_cov_val
             cov_val = np.random.normal(cov_mean, std_val, n_cluster)

             # GENERATE A RANDOM DISTRIBUTION OF POINTS WITHIN THE CLUSTERS

             # Each diagonal element of the cov_matrix is equal to  n millimeter squared (mm²) which signifies a variance of n millimeter for the corresponding variable.
             # This means, on average, data points for that variable deviate from the mean by 1 millimeter. (can be tought as the size of the inner sphere that acts as center
             # in which is centered the normal distribution from which the point are drawn
             mean = int(M/n_cluster)
             std = 100

             # Points assigment
             point_cluster = np.random.normal(mean, std, n_cluster).astype(int)
             print('point cluster initial', point_cluster)

             sum_point_cluster = np.sum(point_cluster)
             if sum_point_cluster != M:

                 if sum_point_cluster < M:

                     difference = M - sum_point_cluster

                     k = 0
                     for i in range(difference):
                         point_cluster[k] = point_cluster[k] + 1
                         k = k+1
                         if k == n_cluster-1:
                             k = 0
                 else:
                     difference = sum_point_cluster - M

                     k = 0
                     for i in range(difference):
                         point_cluster[k] = point_cluster[k] - 1
                         k = k+1
                         if k == n_cluster-1:
                             k = 0

             points = np.zeros((1, 3))

         # à Randomize also the covariance matrix.
             i = 0  # This variable allows the code to gain the corret amount of points for the selected cluster

             for center in centers:
                 c = cov_val[i]

                 cov_matrix = [[c, 0, 0], [0, c, 0],
                               [0, 0, c]]  # Spherical symmetry

                 gmm = multivariate_normal(mean=center, cov=cov_matrix)

                 # Generate random points from the distribution
                 p = gmm.rvs(size=point_cluster[i])
                 points = np.vstack((points, p))
                 i = i+1

             points = np.delete(points, 0, axis=0)
             # CHECK AND ADJUST  points that has crossed the borders

             # If condition fullfilled than place the point some where else in the volume
             # Define volume boundaries (example: cuboid)
             borders = [width, depth, height]
             for i in range(len(points)):
                 for j in range(3):  # Check each dimension

                     condition3 = points[i, j] < 0
                     condition4 = points[i, j] > borders[j]
                     while np.any(condition3) or np.any(condition4):
                         points[i, :] = np.random.rand()*width, np.random.rand()*depth, np.random.rand()*height
                         condition3 = points[i, j] < 0
                         condition4 = points[i, j] > borders[j]

                 # CHECK FOR SUPERIMPOSITION
                 # condition1 = np.sqrt(np.power(points[:i,0]-points[i,0],2)+np.power(points[:i,1]-points[i,1],2)+np.power(points[:i,2]-points[i,2],2)) < r_soma
                 # condition2 = np.sqrt(np.power(points[i:,0]-points[i,0],2)+np.power(points[i:,1]-points[i,1],2)+np.power(points[i:,2]-points[i,2],2)) < r_soma

                     # If condition fullfilled than place the point some where else in the volume
                     while (np.sqrt(np.power(points[:i, 0]-points[i, 0], 2)+np.power(points[:i, 1]-points[i, 1], 2)+np.power(points[:i, 2]-points[i, 2], 2)) < r_soma).any() or (np.sqrt(np.power(points[i+1:, 0]-points[i, 0], 2)+np.power(points[i+1:, 1]-points[i, 1], 2)+np.power(points[i+1:, 2]-points[i, 2], 2)) < r_soma).any():

                         p = np.random.rand()*width, np.random.rand()*depth, np.random.rand()*height

                         points[i, :] = p

             # THIS PART OF CODE COPE WITH THE POSITIONING IN THE LAST CLUSTER OF ALL THE INTERNEURONS: INs (FOR HOW THE CODE IS BUILT)
             # Randomly are generated indicies between the range of the number of neurons and put the correspondent neurons at the
             # bottom so that at the end they will be treated as INs

             Ne = M * excitatory_persentage
             Ni = M-Ne

             IN_indices = set()
             while len(IN_indices) < Ni:
                 index = np.random.randint(0, M-1)
                 IN_indices.add(index)

             # Convert the set of indices to a NumPy array
             IN_indices = np.array(list(IN_indices))
             #print('IN idx',IN_indices)

             # Sort the random indices in descending order for efficient placement
             IN_indices = IN_indices[::-1]

             # FOR EFFICIENCY
             # Column names (optional)
             column_names = ["X", "Y", "Z"]

             # Create DataFrame
             Points_IN = pd.DataFrame(
                 points, columns=column_names).loc[IN_indices]

             # Delete rows using boolean indexing (modifies original DataFrame)
             df = pd.DataFrame(points, columns=column_names)
             Points_EX = df[~df.index.isin(IN_indices)]

             Final_Point = pd.concat([Points_EX, Points_IN], axis=0)

             points_n = []

             points_n = tuple(
                 [list(row) for row in Final_Point.itertuples(index=False, name=None)])

             X = []
             Y = []
             Z = []

             for i in range(len(points)):
                 X.append(points_n[i][0])
                 Y.append(points_n[i][1])
                 Z.append(points_n[i][2])

         elif Type == 1:
             # place neurons non overlapping on area first value of the array
             # The neurons will be placed in a volume slightly lower than the input provided, for sake of simplicity
             
             n_width = width-delta 
             n_depth = depth-delta 
             n_height = height-delta  

             # Just remuve n_
             X, Y, Z = np.zeros((3, M))
             X[0] = random.uniform(delta,n_width)
             Y[0] = random.uniform(delta,n_depth)
             Z[0] = random.uniform(delta,n_height)
             for i in range(1, M):

                 X[i] = random.uniform(delta,n_width)
                 Y[i] = random.uniform(delta,n_depth)
                 Z[i] = random.uniform(delta,n_height)

                 ######
                 # NO OBSTACLES
                 #r,c = get_cell(X[i], Y[i], cell_width, cell_height)

                 ######
                 # or get_H(r,c,H) < 0 no ob
                 while np.any(np.sqrt(np.power(X[:i]-X[i], 2)+np.power(Y[:i]-Y[i], 2)+np.power(Z[:i]-Z[i], 2)) < r_soma):
                     X[i] = random.uniform(delta,n_width)
                     Y[i] = random.uniform(delta,n_depth)
                     Z[i] = random.uniform(delta,n_height)
                     # r,c = get_cell(X[i], Y[i], cell_width, cell_height)  no ob.
             # Storing in the class instance the variables
         
         elif Type == 3:
             
             centers_1s= np.array([[50, 150, 150],
                          [150, 50, 150],
                          [150, 150, 50],
                          [150, 150, 150],
                          [150, 150, 250],
                          [150, 250, 150],
                          [250, 150, 150]])
                 # For this purpose it can be used the array with indicies and multiply them per 10. each mm is discretized by 1000 points
                 #
                
                 # for sake of coherence /100
             centers_1s = [[x/100 for x in row] for row in centers_1s]
             
                 # Convert the list of lists to a NumPy array
             centers_1s = np.array(centers_1s)
                 
                 ### numpy array 140x3 rows are the centers of the smaller custers.
                 
                 
             centers_2s = np.load("C:/Users/leona/Desktop/Master thesis/PY/Network_class/Fractals/centers_2p.npy")
             centers_2s = [[x/1000 for x in row] for row in centers_2s]
                 
                 
             centers_2s = np.array(centers_2s)

                 ## Now is needed to cope with che distribution of the points.
                 # For the explanations and assumptions made for each appreoache consult the notes on tablet
                  
             centers = np.concatenate([centers_1s, centers_2s],axis=0)
             
             
                 #### 2ND APPROACH  ####
             F  = int(M/27 )  
             n_1s = F*7
             n_2s = M-n_1s
             points = np.zeros((1,3))
                 
           
                 
                 
                 
             cov = [[cov1_fractals, 0, 0], [0, cov1_fractals, 0], [0, 0, cov1_fractals]]  # Spherical symmetry
             cov_2s = [[cov2_fractals , 0, 0], [0, cov2_fractals , 0], [0, 0, cov2_fractals ]]
                 
             mean_1s = int(n_1s/(centers_1s.shape[0]))
             mean_2s = int(n_2s/(centers_2s.shape[0]))
             std = 5
     
             point_cluster_1s = np.random.normal(mean_1s, std,centers_1s.shape[0]).astype(int);
             point_cluster_2s = np.random.normal(mean_2s, std,centers_2s.shape[0]).astype(int);
                 
                 
             point_cluster = np.concatenate([point_cluster_1s,point_cluster_2s],axis=0)
             
             sum_point_cluster = np.sum(point_cluster)
             if sum_point_cluster != M:

                 if sum_point_cluster < M:

                     difference = M - sum_point_cluster

                     k = 0
                     for i in range(difference):
                         point_cluster[k] = point_cluster[k] + 1
                         k = k+1
                         if k == n_cluster-1:
                             k = 0
                 else:
                     difference = sum_point_cluster - M

                     k = 0
                     for i in range(difference):
                         point_cluster[k] = point_cluster[k] - 1
                         k = k+1
                         if k == n_cluster-1:
                             k = 0

             points = np.zeros((1, 3))
             
             i = 0 ## This variable allows the code to gain the corret amount of points and covariance matrix for the selected cluster
                 
             for center in centers:
                     ## The first 7 points represent the major cluster centers 
                    if i <= 7:
                         gmm = multivariate_normal(mean=center, cov=cov)
                     
                     # Generate random points from the distribution
                         p = gmm.rvs(size = point_cluster[i])
                         #print('POINTS',p)
                         points = np.concatenate([points,p],axis = 0)
                         i = i+1
                    else:
                         #print('POINTS',p)
                         gmm = multivariate_normal(mean=center, cov=cov_2s)
                       
                       # Generate random points from the distribution
                         p = gmm.rvs(size = point_cluster[i])
                         points = np.concatenate([points,p],axis = 0)
                         i = i+1
                 
                     
                         
                 # Ruling out the first row       
             points = np.delete(points, 0, axis=0)
                  
                 
                 
                 #print('POINTS',points)
                  ## CHECK AND ADJUST  points that has crossed the borders
                  
                  ## If condition fullfilled than place the point some where else in the volume
                  # Define volume boundaries (example: cuboid)
             borders = [width,depth,height]
             check = 0
             for i in range(len(points)):
                     for j in range(3):  # Check each dimension
                         
                         condition3 = points[i,j]<0
                         condition4 = points[i,j]>borders[j]
                         while np.any(condition3) or np.any(condition4):
                             points[i,:] = np.random.rand()*width,np.random.rand()*depth,np.random.rand()*height
                             condition3 = points[i,j]<0
                             condition4 = points[i,j]>borders[j]
                             
                     ## CHECK FOR SUPERIMPOSITION        
                     # condition1 = np.sqrt(np.power(points[:i,0]-points[i,0],2)+np.power(points[:i,1]-points[i,1],2)+np.power(points[:i,2]-points[i,2],2)) < r_soma
                     # condition2 = np.sqrt(np.power(points[i:,0]-points[i,0],2)+np.power(points[i:,1]-points[i,1],2)+np.power(points[i:,2]-points[i,2],2)) < r_soma
                         
                         
                         while  (np.sqrt(np.power(points[:i,0]-points[i,0],2)+np.power(points[:i,1]-points[i,1],2)+np.power(points[:i,2]-points[i,2],2)) < r_soma).any() or (np.sqrt(np.power(points[i+1:,0]-points[i,0],2)+np.power(points[i+1:,1]-points[i,1],2)+np.power(points[i+1:,2]-points[i,2],2)) < r_soma).any(): ## If condition fullfilled than place the point some where else in the volume
                             
                             p =  np.random.rand()*width,np.random.rand()*depth,np.random.rand()*height
                             
                             points[i,:] = p
                             check =+ 1
                             
                             
             Ne = M * excitatory_persentage
             Ni = M-Ne

             IN_indices = set()
             while len(IN_indices) < Ni:
                 index = np.random.randint(0, M-1)
                 IN_indices.add(index)

             # Convert the set of indices to a NumPy array
             IN_indices = np.array(list(IN_indices))
             #print('IN idx',IN_indices)

             # Sort the random indices in descending order for efficient placement
             IN_indices = IN_indices[::-1]

             # FOR EFFICIENCY
             # Column names (optional)
             column_names = ["X", "Y", "Z"]

             # Create DataFrame
             Points_IN = pd.DataFrame(
                 points, columns=column_names).loc[IN_indices]

             # Delete rows using boolean indexing (modifies original DataFrame)
             df = pd.DataFrame(points, columns=column_names)
             Points_EX = df[~df.index.isin(IN_indices)]

             Final_Point = pd.concat([Points_EX, Points_IN], axis=0)

             points_n = []

             points_n = tuple(
                 [list(row) for row in Final_Point.itertuples(index=False, name=None)])

             X = []
             Y = []
             Z = []

             for i in range(len(points)):
                 X.append(points_n[i][0])
                 Y.append(points_n[i][1])
                 Z.append(points_n[i][2])
            
             
             
             
             
             
             
             
         self.X = X
         self.Y = Y
         self.Z = Z

         return X, Y, Z
    
     def grow_W(self,Number_GABA,Number_GLU,bias_GABA,bias_GLU,alpha_values,Gamma_GABA,Gamma_GLU,P_b,Dl,Branch_TH):
        
         
        ######### DESCRIPTION
        
        
        
        ########### INHERETED PARAMETERS
        excitatory_persentage = self.excitatory_persentage 
        
        
        
        
        # UNINDEND ###########################
        Vector_Field = self.Vector_Field
        over_factor = self.over_factor
        
        
        
        
        X = self.X
        Y = self.Y
        Z = self.Z
        
        
        width = self.width     # culture width (mm)
        height = self.height    # culture height (mm)
        depth = self.depth
        
        ########### Initialize variables
        
        N_neurons = len(X)
        Ne = int(N_neurons*excitatory_persentage)
        Ni = N_neurons-Ne
        
        # Total Number of neurites
        N_neurites = Ne*Number_GLU + Ni*Number_GABA
        
        
        # default arguments # could be changed
        Dl      = Dl # axon segment length (mm)
        
        ## Greater the value winder the path
        phi_sd_GLU = 0.05  # axon random walk std ### ORIGINAL VALUES 0.1
        theta_sd_GLU = 0.05 # second angle
        
        phi_sd_GABA = 0.25   
        theta_sd_GABA = 0.25 

        r_dendrite_mu   = 120e-3    # denrite radius mean (mm) ## Depends on the cells' type M1 and M2 aprox. 117
        r_dendrite_sd   = 40e-3     # denrite radius std
        
        r_dendrite = r_dendrite_mu + r_dendrite_sd*np.random.normal(0,1,N_neurons)
        
        A = np.zeros((N_neurons,N_neurons))
        # This variable keeps track which are the indexes of the collateral neurites for each neuron in growing variables (numpy array of lists)
        # The second one stores the neuron of belonging of the collateral in the corresponding index in the growing variables
        # Xi,Yi,Zi
        Neurites_Neuron_idx = np.array([N_neurons]) 
        
        # The first indicies can be easily pre-allocated because are the 'main' processes of the neurons.
          # Empty integer array
        
        Neuron_of_Neurites_glu = np.tile(np.arange(0,Ne), Number_GLU)        
        Neuron_of_Neurites_gaba = np.tile(np.arange(Ne,Ne+Ni), Number_GABA)
      
    
        # Neuron of neurite ids are strictly coupled with the ones of Total_length and Xi,Yi,Zi
        Neuron_of_Neurites_idx = np.concatenate((Neuron_of_Neurites_glu,Neuron_of_Neurites_gaba))
        
        
        # The variable is an array of lists. Row-wise the idx refer to the neurons (pre-synaptic )meanwhile the 
        # first column stores the info of which are the neurons synaptically connected (post-synaptic)
        
        
        Connections = np.arange(N_neurons,dtype = object)
        
        
        
        
        
        
        
        
        
        # Use cKDTree for efficient nearest neighbor search
        
        # UNINDEND ###########################
        Vector_field_surrogate = self.Vector_field_surrogate
        
        # The position points must be scaled with regard to Dl to obtain the same measure scale than 
        # the points
        Vector_field_tree = cKDTree(self.Vector_field_surrogate)
        
        
        
        
        
        
        
        
        
        
        # Initialize for each neuron the total neurite length. Each branch length will be defined a priori and drawn 
        # out from a Raylaight distribution and its length subtracted from the total length available. Another approach 
        # would be to introduce a further term that represents the probability at each step that the branch ceases growing.
        
        # TOTAL LENGTH of the neurons' neurites. Change for neurons' duplicates
        Total_length_GLU = np.squeeze(np.random.rayleigh(scale=0.18, size=Ne*Number_GLU)+(bias_GLU)) #[mm]  
        
        Total_length_INH = np.squeeze(np.random.rayleigh(scale=0.2, size=Ni*Number_GABA)+(bias_GABA)) #[mm] # total number of main branches
        
        # Concatenate. The idx are matched with the ones of Neuron_of_Neurites_idx
        Total_length = np.append(Total_length_GLU, Total_length_INH) # MAIN 
        
        # Convert in number of segments
        Total_length = np.squeeze(Total_length/Dl)
        print('Total length:',Total_length)
        # Calculate the maximum number of segments (maybe not necessary)
        Ns = int(np.max(Total_length/Dl))
        N_branch = Ne*Number_GLU+Ni*Number_GABA
        
        # STARTING POINTS Array  ######## CORRECTLY DEFINE THE STARTING POINTS IN BASE OF THE NUMBER OF NEURITES
        # Initialize the starting coordinates.
        # Glutamatergic
        X_GLU = np.tile(X[0:Ne], Number_GLU)
        Y_GLU  = np.tile(Y[0:Ne], Number_GLU)
        Z_GLU  = np.tile(Z[0:Ne], Number_GLU)
        
        X_GABA = np.tile(X[Ne:], Number_GABA)
        Y_GABA  = np.tile(Y[Ne:], Number_GABA)
        Z_GABA  = np.tile(Z[Ne:], Number_GABA)
        
        
        # We want to keep track
        Xi = np.zeros((N_branch,Ns))
        Yi = np.zeros((N_branch,Ns))
        Zi = np.zeros((N_branch,Ns))
        
        
        
        
        # These variables will store continuosly all the growing neurites' tip as well as the last tip of inactive neurites
        Xi[:,0] = np.concatenate((X_GLU,X_GABA))
        Yi[:,0] = np.concatenate((Y_GLU,Y_GABA))
        Zi[:,0] = np.concatenate((Z_GLU,Z_GABA))
        
        
        
        
        # Xj = np.concatenate((X_GLU,X_GABA))
        # Yj = np.concatenate((Y_GLU,Y_GABA))
        # Zj = np.concatenate((Z_GLU,Z_GABA))
        
        
        # MxNs matricies rows are the neurons, columns are the locations of each neurite's segment for that dimension
        # In Total_length neurons' indicies are encoded in the row of belonging
        
        # X_a =  np.zeros((N_neurites,np.max(Ns)), dtype=float)
        # Y_a =  np.zeros((N_neurites,np.max(Ns)), dtype=float)
        # Z_a =  np.zeros((N_neurites,np.max(Ns)), dtype=float)
        
        
        ###### START THE GROWTH
        
        # Initial values
        # X_a[:, 0] = Xi
        # Y_a[:, 0] = Yi
        # Z_a[:, 0] = Zi
        # 
        ## This expression generates a random number between 0 and 2π radians by first generating 
        ## a random number between 0 and 1 using np.random.rand, then multiplying it by 2π.
        
        phi = 2*np.pi*np.random.rand(N_neurites) # Azimuthal angle [0 360]
        theta = np.pi*np.random.rand(N_neurites) # Polar angle [0 180]
        
        
        '''
        Spherical <-> cartesian frames
        
        x = r sin(theta) cos(phi)
        y = r sin(theta) sin(phi)
        z = r cose(theta)
        
        phi
        
        theta = arccos(z/sqrt(x**2 + y**2 + z**2))
        phi = sgn(y) arccos(z/sqrt(x**2 + y**2))
                    
        
        '''
        
        
        
        
        ### MAIN LOOP ###
        # # Initializes the number of growing cones, TO BE UPDATED EVERY TIME A BRANCH IS CREATED
        # Number_cones = len(X)
        n = 1
        Finished_neurons = 0
        # To keep track of the progress
        
        # The loop runs until every single number in the Total_length array becomes negative
        # (less than 0).  If even one number in the array is zero or positive, the loop keeps running.
        while not  np.all(Total_length <= 0): 
            
            
            if len(np.where(Total_length<=0)[0]) != Finished_neurons:
                
                Finished_neurons = len(np.where(Total_length<=0)[0])
                print('Fully extented neurons:')
                print(Finished_neurons)
            
           
            # Check which neurons must be lengthen.
            
            # Mask_growth = np.where(Total_length > 0)
            Mask_growth= Total_length > 0
            
            # Retrieve the info about position of collateral neurites per neuron in the growing variables Xi,Yi,Zi
            
                       
            
            # determine axon growth direction:
            # The condition to construct the mask regards which neuron has still length count different from zero.
            Dx = Dl*np.cos(phi)*np.sin(theta)
            Dy = Dl*np.sin(phi)*np.sin(theta)
            Dz = Dl*np.cos(theta)
            
            
            # Need to update this
            Xi[Mask_growth,n] = Xi[Mask_growth,n-1]+Dx[Mask_growth]
            Yi[Mask_growth,n] = Yi[Mask_growth,n-1]+Dy[Mask_growth]
            Zi[Mask_growth,n] = Zi[Mask_growth,n-1]+Dz[Mask_growth]
            
            
            ########### 
            
            
            Pre_point = np.transpose(np.array([Xi[Mask_growth,n-1], Yi[Mask_growth,n-1], Zi[Mask_growth,n-1]]))
            
            Temp_point = np.transpose(np.array([Xi[Mask_growth,n], Yi[Mask_growth,n], Zi[Mask_growth,n]]))
            
            Neuron_of_belonging = Neuron_of_Neurites_idx[Mask_growth]
            
            
            
            #TODO: CHECKK
            # ------- IN CASE BORDER CHECK AND VECTOR FIELD ARE COMMENTED
            # Final_points = Temp_point
            
            ###########  CHECK FOR BORDER CROSSING: parallelized
            
            Final_points,phi_new,theta_new = Check_borders(Temp_point,Pre_point,phi[Mask_growth],theta[Mask_growth],width,depth,height,Dl)
            phi[Mask_growth] = phi_new
            theta[Mask_growth] = theta_new
            
            ###########  VECTOR FIELD INFLUENCE: parallelized
            
            
            # Final_points,phi_new,theta_new = Field_gradient(Vector_field_tree,Vector_field_surrogate,Vector_Field,Pre_point,Final_points,over_factor,Dl, Connections, Neuron_of_belonging,phi[Mask_growth],theta[Mask_growth])
           
            
            # phi[Mask_growth] = phi_new
            # theta[Mask_growth] = theta_new
           
            
            
            ###########  CONNECTIONS:
                
            A , Connections = Check_synapses(X,Y,Z,A,Final_points,Connections,alpha_values,r_dendrite,Neuron_of_belonging,Ne)
            
        
            # Update Xj,Yj,Zj. Along a new iteration these varibles will become at the end the new Xi,Yi and Zi. Adding the Dx Dy and Dz 
            # quantities the new XjYjand Zj variables are calculated
            
            
            # UNINDEND ###########################
            Xi[Mask_growth,n] = Final_points[:,0] # First column
            Yi[Mask_growth,n] = Final_points[:,1]
            Zi[Mask_growth,n] = Final_points[:,2]
            
            
            
            
            
            
            # Update all total lengths
            
            Total_length[Mask_growth] = Total_length[Mask_growth]-1
                
            ###########  BRANCHING:  
                
            # Generate randomly number between 0 and 1. The vector size matches the currently growing trees
                       
            rdn = np.random.uniform(0,1,len(Xi[Mask_growth]))
            
            # Extract the indicies of the growing branches; it allows us to determine the real branch
            # idx that undergoes splitting
            Idx_growing_branches = np.where(Mask_growth)[0]
            
            indices = np.where(rdn <= P_b)[0]
            
            if indices.size != 0:
                print('')
                print('----------------------------------------------------------------------')
                print('')
                print(f'{len(indicies)} new branches')
                print(f'Total number of brances: {len(Idx_growing_branches) + len(indicies)}')
                print('')
                print('----------------------------------------------------------------------')
                print('')
                for ind_branch in indices:
                    
                   
                                     
                   
                    
                    
                    # Extract all the necessary variables
                    
                    Parent_branch_idx = Idx_growing_branches[ind_branch]
                    
                    Parent_branch_neuronal_type = Neuron_of_Neurites_idx[Parent_branch_idx]
                    
                    # Check the length... if minor than a TH just skip it 
                    Total_length_parent = Total_length[Parent_branch_idx] # In growing segments
                    # The length of the branch is taken from the a raylight distribution which scale parameter is set to obtain all the values generated within 0 and 1
                    # and then linearly map in a range of values between 0 and the maximum length still available of the parent branch
                    
                
                    
                    random_data = np.random.rayleigh(scale=0.4, size=1)
                    branch_length = map_range(random_data, 0, 1, 0, Total_length_parent) # parent length is arelady expressed in terms of Dl
                    print('New Branch!, Length:',branch_length)
                    if branch_length<Branch_TH/Dl:
                        print('BBB')
                        continue
                    
                    # # Parent branch's final point. Used to define the new branch's direction
                   
                    Phi_parent = phi[Parent_branch_idx]
                    Theta_parent = theta[Parent_branch_idx]
                    
                    
                    X_parent = Xi[Parent_branch_idx,n-1]
                    Y_parent = Yi[Parent_branch_idx,n-1]
                    Z_parent = Zi[Parent_branch_idx,n-1]
                    
                    ########FROM HERE
                    # New segment's coordinates
                    # Generate new angles and add to the parent's relative angle
                    phi_branch = Phi_parent + generate_random_angles() # """Generates a random angle from the combined range [pi/5, pi] and [-pi, -pi/5]."""
                    theta_branch =  Theta_parent + generate_random_angles()
                    
                    # print('Phi Theta',[math.degrees(phi_branch), math.degrees(theta_branch)])
                    # Add the new angles in the variabbles
                    phi = np.append(phi,phi_branch)
                    theta = np.append(theta,theta_branch)
                    
                    
                    Dx_branch = Dl*np.cos(phi_branch)
                    Dy_branch = Dl*np.sin(phi_branch)
                    Dz_branch = Dl*np.sin(theta_branch)
                    
                    X_new = X_parent+Dx_branch
                    Y_new = Y_parent+Dy_branch
                    Z_new = Z_parent+Dz_branch
                    
                    # Append a new vector in Xi,Yi and Zi
                    Xi = np.vstack([Xi, np.zeros(Ns)])
                    Yi = np.vstack([Yi, np.zeros(Ns)])
                    Zi = np.vstack([Zi, np.zeros(Ns)])
                    
                    Xi[-1,n] = X_new
                    Yi[-1,n] = Y_new
                    Zi[-1,n] = Z_new
                    
                    # Add the new branch positions
                  
                    # Update parent culture's total length
                    Total_length[Parent_branch_idx] = Total_length_parent-branch_length
                    
                    # Insert new branch in all the necessary variables
                    Total_length = np.append(Total_length,branch_length)
                    Neuron_of_Neurites_idx = np.append(Neuron_of_Neurites_idx,Parent_branch_neuronal_type)
                    
                    # Add to Xj,Yj,Zj the new branch tip's coordinates: X_new,Y_new,Z_new  Along a new iteration these varibles will become at the end the new Xi,Yi and Zi. Adding the Dx Dy and Dz 
                    # quantities the new XjYjand Zj variables are calculated
                 
                    # Xj = np.append(Xj,X_new)
                    # Yj = np.append(Yj,X_new)
                    # Zj = np.append(Zj,X_new)
                    #
                    
        
        
        
            # Xi = Xj
            # Yi = Yj
            # Zi = Zj
             
            
            # The new angles must be addedd in compliance with the type of neurons involved.
            # Find which neurons in xi yi and zi are glu and which gaba and address them with the 
            # respective std
            
            # Find in Neuron_of_Neurites_idx the idx of GLU neurons => neurons id <=Ne
            # and the inhibitory idx => neuronsid > Ne
            
            idx_GLU,idx_GABA = find_indices(Neuron_of_Neurites_idx, Ne)
         
                
                
            phi[idx_GLU] += phi_sd_GLU*np.random.normal(0, 1, len(Xi[idx_GLU]))
            theta[idx_GLU]  += theta_sd_GLU*np.random.normal(0, 1, len(Xi[idx_GLU]))
            
            phi[idx_GABA] += phi_sd_GABA*np.random.normal(0, 1, len(Xi[idx_GABA]))
            theta[idx_GABA]  += theta_sd_GABA*np.random.normal(0, 1, len(Xi[idx_GABA]))
            
            
            n = n+1
         
        return A,Xi,Yi,Zi,Connections
 
