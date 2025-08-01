import random
import math

from scipy.stats import skewnorm
from scipy.stats import chi2
import seaborn as sns
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


def Check_borders(Point_neurite_temp_vector,Point_neurite_pre_vector,Phi,Theta,width,depth,height,Dl) :
    '''
    Checks for the neurites branches that crosses the culture's boundaries.
    If the point crosses x and y boundaries than the algorithm will act only on
    the azhimutal angle, if z axes boundaries are crossed than the polar angle is changed.
    
    

    Parameters
    ----------
    Neurites related variables have been altrady masked
    
    Point_neurite_temp_vector : array [N,3]
        Temporary new points of the growing neurites  
    Point_neurite_pre_vector : array [N,3]
        Current points of the growing neurites 
    Phi : array [N,3]
        Azimuthal angles 
    Theta : array [N,3]
        Polar angles
    width : Integer
        X span of the culture
    depth : Integer
        Y span of the culture
    height : Integer
        Z span of the culture
    Dl : Float 
        Growing step

    Returns
    -------
    Point_neurite_temp : array [N,3]
        New temporary points of the growing neurites  
    phi : array [N,3]
        New azimuthal angles 
    theta : array [N,3]
        New polar angles
    '''
    
    
    
    z_axes = np.array((2,5))
    xy_axes = np.array((0,1,3,4))
    sd = 0.1 # Is needed to bring some randomness in the debug, otherwise it gets stuck
    debug_angle = np.pi/180 # Used to avoid bottle necks in the search of a point. pi/96 = 1.8334649 degrees
            # in compliance with the boundary conditions.
    # t_th = 100 # Number of iterations before getting in the debug loop
    p_add = 0.002 # The addition to the multiplier
    # Iterate over neurites
    for neu in np.arange(np.shape(Point_neurite_temp_vector)[0]):
        
        # Extract variables
        Point_neurite_temp = Point_neurite_temp_vector[neu,:]
        Point_neurite_pre = Point_neurite_pre_vector[neu,:]
        phi = Phi[neu]
        theta = Theta[neu]
        Ch = False
        
        
        # Set conditions
        exceed_threshold = [Point_neurite_temp[0] < 0, Point_neurite_temp[1] < 0, Point_neurite_temp[2] < 0, Point_neurite_temp[0]> width, Point_neurite_temp[1] > depth, Point_neurite_temp[2] > height]
        Check = any(exceed_threshold)
        phi_ = phi
        theta_ = theta
        
        if Check == True:
            Ch = True
            # Used to avoid bottlenecks
            t = 0    
            p = 0
            q = sd*np.random.normal(0, 1) # Needed to add some variability
            f = 1
            
            # Axes of incidence
            Exceeded_Axes = np.where(exceed_threshold)[0]
            
            
            if np.all(np.isin(Exceeded_Axes, xy_axes)): # Act only on the azimuthal angle phi
                # print('xy only')   
                # print('Phi_pre', np.degrees(phi))
                while Check == True:
                    
                    
                    # There are situation in whichh the algorithm is stuck in a loop reaching t = 80000.
                    # if t == t_th :  # Attention to the trade off ruled by the term
                    # print('Inside debug')
                    f = f*(-1)
                    q = q + p*(f)# At every iteration the value changes sign and takes, thus the opposite magnitude plus one
                    # change completely tha angle both of them but an improvement could be to find which axes is the problematic one and solve
                    # Another way to debug is to determine the plane that is crossed and rotate away the related angle.
                    
                    phi_ = phi + q*debug_angle # Rotation direction depends on q's sign
                    
                    p = p+p_add
                    # print('P',q*debug_angle )
                        # t = 0
                    # else:
                        
                        
                    
                        
            
                    # i-th elements shuld be preserved not modified ###### in case check
                    Point_neurite_temp[0] = Point_neurite_pre[0] + Dl*np.cos(phi_)*np.sin(theta)
                    Point_neurite_temp[1] = Point_neurite_pre[1] + Dl*np.sin(phi_)*np.sin(theta)
                    Point_neurite_temp[2] = Point_neurite_pre[2] + Dl*np.cos(theta)
                    
                    
            
                    exceed_threshold = [Point_neurite_temp[0] < 0, Point_neurite_temp[1] < 0, Point_neurite_temp[2] < 0, Point_neurite_temp[0]> width, Point_neurite_temp[1] > depth, Point_neurite_temp[2] > height]
                    
                    Check = any(exceed_threshold)
                    
                    t = t+1  # For computation feasability
                    
            
                    
            elif np.all(np.isin(Exceeded_Axes, z_axes)): # Act on the polar angle theta
                # print('z only') 
                # print('Theta_pre', np.degrees(theta))
                while Check == True:
                    
                    
                    # There are situation in whichh the algorithm is stuck in a loop reaching t = 80000.
                    # if t == t_th :  # Attention to the trade off ruled by the term
                    # print('Inside debug')
                    f = f*(-1)
                    q = q + p*(f)# At every iteration the value changes sign and takes, thus the opposite magnitude plus one
                    # change completely tha angle both of them but an improvement could be to find which axes is the problematic one and solve
                    # Another way to debug is to determine the plane that is crossed and rotate away the related angle.
                    
                    
                    theta_ = theta +  q*debug_angle # Rotation direction depends on q's sign
                    p = p+p_add
                    # print('P',q*debug_angle )
                    # t = 0
                    # else:
                        
                        
                        
                    
            
                    # i-th elements shuld be preserved not modified ###### in case check
                    Point_neurite_temp[0] = Point_neurite_pre[0] + Dl*np.cos(phi)*np.sin(theta_)
                    Point_neurite_temp[1] = Point_neurite_pre[1] + Dl*np.sin(phi)*np.sin(theta_)
                    Point_neurite_temp[2] = Point_neurite_pre[2] + Dl*np.cos(theta_)
            
                    exceed_threshold = [Point_neurite_temp[0] < 0, Point_neurite_temp[1] < 0, Point_neurite_temp[2] < 0, Point_neurite_temp[0]> width, Point_neurite_temp[1] > depth, Point_neurite_temp[2] > height]
                    
                    Check = any(exceed_threshold)
                    
                    t = t+1  # For computation feasability
            
            else: # both xy and z are crossed
                # print('x/y and z')   
                # print('Phi_pre', np.degrees(phi))
                # print('Theta_pre', np.degrees(theta))
                while Check == True:
                    
                    
                    # There are situation in whichh the algorithm is stuck in a loop reaching t = 80000.
                    # if t == t_th :  # Attention to the trade off ruled by the term
                    # print('Inside debug')
                    f = f*(-1)
                    q = q + p*(f)# At every iteration the value changes sign and takes, thus the opposite magnitude plus one
                    # change completely tha angle both of them but an improvement could be to find which axes is the problematic one and solve
                    # Another way to debug is to determine the plane that is crossed and rotate away the related angle.
                    
                     
                    phi_ = phi + q*debug_angle # Rotation direction depends on q's sign
                    theta_ = theta +  q*debug_angle
                    p = p+p_add
                    # print('P',q)
                    # t = 0
                    # else:
                        
                        
                    phi+= sd*np.random.normal(0, 1)
                    theta  += sd*np.random.normal(0, 1)
            
                    # i-th elements shuld be preserved not modified ###### in case check
                    Point_neurite_temp[0] = Point_neurite_pre[0] + Dl*np.cos(phi_)*np.sin(theta_)
                    Point_neurite_temp[1] = Point_neurite_pre[1] + Dl*np.sin(phi_)*np.sin(theta_)
                    Point_neurite_temp[2] = Point_neurite_pre[2] + Dl*np.cos(theta_)
            
                    exceed_threshold = [Point_neurite_temp[0] < 0, Point_neurite_temp[1] < 0, Point_neurite_temp[2] < 0, Point_neurite_temp[0]> width, Point_neurite_temp[1] > depth, Point_neurite_temp[2] > height]
                    
                    Check = any(exceed_threshold)
                    
                    t = t+1  # For computation feasability
            
                
           
        # if Ch == True:    
        #     print('Phi_post', np.degrees(phi_))
        #     print('Theta_post', np.degrees(theta_))
        # # Update variables
        Point_neurite_temp_vector[neu,:] = Point_neurite_temp
        Phi[neu] = phi_ # azimuthal
        Theta[neu] = theta_
    
    return Point_neurite_temp_vector,Phi,Theta


def find_approximate_match(number, vector, tolerance=0.4):
    """
    Finds the indices of values in the vector that are within the tolerance of the given number.

    Args:
        number: The number to find approximate matches for.
        vector: The vector of numbers to search.
        tolerance: The maximum allowed deviation between the number and the vector value.

    Returns:
        A list of indices where approximate matches were found.
    """
    matches = []
    for i, value in enumerate(vector):
        if abs(number - value) <= tolerance:
            matches.append(i)
    return matches


def get_dendrite_prob(r_1,r_0,dx,map_magnitude):
    '''
    This function is thought to establish a evidence-based distance-dependent 
    connection probability field about the interested somata within the shell enacapsuled
    in 'r_0' and 'r_1', this whithout a detailed representation of the dendritic 
    arbourization. The analysis hinges on the morphometric analysis on hipsc control lines
    in the following papers:
        1) https://doi.org/10.1038/s41467-019-12947-3
        2) https://doi.org/10.1016/j.celrep.2020.107538
    
    The sholl analysis evidenced a peak in branching phenomena at approx. 40/50 um from the
    soma. Therefore the switch of neurites from primary branches to secondary ones is likely
    to happen here. This is important to set the correct neurite diameter 'Branch_d'. The mean
    dendritic length distribution across the distance from the soma is reproduced by scaling
    a Chi-squared distribution (along both axes) with 4 degrees of freedom.
    The purpose is to determine the ratio between the total volume of dendritic abourization 
    within the shell and the total volume of the latter. The total branch length is calculated 
    by computing the cumulative density function between r_0 and r_1 (integral through the trapz 
    function of the PDF within the defined range). In this way the probability of an
    axon to cross a dendritic process is defined. Given the distribution the influence region spans
    approx 200 um in radius even though the probabiities at the boundaries borders on zero.
    
    Parameters
    ----------
    r_1 : float [um]
        Outer shell radius
    r_0 : float [um]
        Inner shell radius
    dx : float 
        Spacing between sample points for the trapezoidal method
    map_magnitude : integer [um]
        Upper limit of the scaling map of the distribution's x axes.
    Returns
    -------
    Filled_volume : float 
        Probability of neurite presence in the shell [0,1]

    '''
    
    if r_1 <= 40:
        Branch_d = 2.1 # [um] # primary neurites
        # r_1 = 40
        # r_0 = 0
        
    else:
        Branch_d = 1.51 # [um] # secondary neurites
        r_1 = r_1
        r_0 = r_0
    
    
    value_ = np.arange(0,250,0.1)  #[um] Perisomatic location
    sqr = np.zeros(len(value_))
    j=0
    for i in value_:
        
        maped_val = map_range(i, 0, map_magnitude, 0, 12)    
        sqr[j] = chi2.pdf(maped_val, 4)
        j=j+1
        

    
    #%

    sqr_max = max(sqr)
    value_ = np.arange(0,250,0.1)  #[um] Perisomatic location
    sqr = np.zeros(len(value_))
    map_sqr = np.zeros(len(value_))
    j=0
    mode = 110
    for i in value_:
        
        maped_val = map_range(i, 0, map_magnitude, 0, 12)    
        sqr = chi2.pdf(maped_val, 4)
        # map_sqr[j] = sqr*714.2
        maped_val_ = map_range(sqr,0, sqr_max,0,mode)    
        map_sqr[j] = maped_val_
        j=j+1
         
    plt.figure()
    plt.title('Scaled Chi-squared distribution. DoF: 4')
    plt.plot(value_,map_sqr)
    plt.xlabel('Distance form soma [um]')
    plt.ylabel('Mean dendritic length [um]')    

    x_v_1= np.where(value_ == r_1)[0][0]
    x_v_0= np.where(value_ == r_0)[0][0]

    x_1 = value_[x_v_0:x_v_1]
    y_1= map_sqr[x_v_0:x_v_1]
    Cumulative_length_1 = np.trapz(y_1,x_1,dx=dx)

    ##### Calculate
     
    V_sphere = (4/3)*np.pi*(r_1**3) -(4/3)*np.pi*(r_0**3)

    
    V_dendrite = (Cumulative_length_1/4)*np.pi*(Branch_d**2)

    Filled_volume = (V_dendrite/V_sphere)
    
    return Filled_volume
    





def map_range(value, from_min, from_max, to_min, to_max):
    """
    Maps a value from one range to another range using linear interpolation.

    Args:
        value: The value to be mapped.
        from_min: The minimum value of the original range.
        from_max: The maximum value of the original range.
        to_min: The minimum value of the target range.
        to_max: The maximum value of the target range.

    Returns:
        The mapped value in the target range.
    """

    # Check for valid input ranges (avoid division by zero)
    if from_max - from_min == 0:
        raise ValueError("Input range cannot be zero.")
    if to_max - to_min == 0:
      raise ValueError("Target range cannot be zero.")

    # Linear transformation formula
    mapped_value = (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min
    return mapped_value




def extract_nonzero_arrays_separately(Xi, Yi, Zi):
    """
    Generates N arrays for each of Xi, Yi, and Zi, containing the non-zero 
    elements from the corresponding rows.

    Args:
        Xi: NumPy array of shape (N, K)
        Yi: NumPy array of shape (N, K)
        Zi: NumPy array of shape (N, K)

    Returns:
        A tuple containing three lists:
        - x_arrays: A list of N NumPy arrays with non-zero elements from Xi.
        - y_arrays: A list of N NumPy arrays with non-zero elements from Yi.
        - z_arrays: A list of N NumPy arrays with non-zero elements from Zi.
        Returns None if the input shapes are incompatible.
    """

    N = Xi.shape[0]
    if Yi.shape[0] != N or Zi.shape[0] != N:
        print("Error: Input arrays must have the same number of rows (N).")
        return None

    x_arrays = []
    y_arrays = []
    z_arrays = []

    for i in range(N):
        x_nonzero = Xi[i][Xi[i] != 0]  # Non-zero elements from Xi's row i
        y_nonzero = Yi[i][Yi[i] != 0]  # Non-zero elements from Yi's row i
        z_nonzero = Zi[i][Zi[i] != 0]  # Non-zero elements from Zi's row i

        x_arrays.append(x_nonzero)
        y_arrays.append(y_nonzero)
        z_arrays.append(z_nonzero)

    return x_arrays, y_arrays, z_arrays





def find_indices(arr, Ne):
    """
    Finds indices of elements in a NumPy array that are less than or equal to Ne,
    and separately, indices of elements that are greater than Ne.

    Args:
        arr: The NumPy array.
        Ne: The threshold value.

    Returns:
        A tuple containing two NumPy arrays:
        - indices_GLU: Indices of elements less than or equal to Ne.
        - indices_GABA: Indices of elements greater than Ne.
    """

    indices_GLU = np.where(arr < Ne)[0]  # Use [0] to get indices from the tuple
    indices_GABA = np.where(arr >= Ne)[0]   # Use [0] to get indices from the tuple

    return indices_GLU, indices_GABA


def remove_matching_rows(matrix, values):
    """
    Removes rows from a matrix where the element in the FOURTH column matches a value in a given array.

    Args:
        matrix: A NumPy 2D array.
        values: A NumPy 1D or 2D array.

    Returns:
        A new NumPy 2D array with the matching rows removed. Returns the original matrix if no matches are found, or an empty array if input dimensions are incorrect.
    """
    # print('matrix',matrix)
    # print('values',values)
    # if matrix.ndim != 2 or values.ndim not in (1, 2):
    #   return np.array([])  # Handle incorrect input dimensions

    _column = matrix[:, 4]  # Get the 5th column (index 4)
    # if values.ndim == 2:
    #   values = values.flatten() # Flatten 2D array to 1D if necessary.

    matches = np.isin(_column, values)  # Find rows with matches

    rows_to_keep = ~matches  # Get the indices of rows to KEEP (not the matches)

    filtered_matrix = matrix[rows_to_keep]  # Select only the rows to keep

    return filtered_matrix



def Get_vectors(Pre_point, Temp_point):
    """
    CHECKED: 1
    
    Calculates the vectors from points Pre_point to points Temp_point and normalizes their magnitudes.

    Args:
        Pre_point: A NumPy array of shape (N, 3) representing the coordinates of points Pre_point .
        Temp_point: A NumPy array of shape (N, 3) representing the coordinates of points Temp_point.

    Returns:
        A NumPy array of shape (N, 3) containing the normalized vectors from Pre_point to Temp_point.
        """

    # print("Shape of Pre_point rom get gector:", Pre_point.shape)
    # print("Shape of Temp_point rom get gector:", Temp_point.shape)
    vectors = Temp_point - Pre_point  # Calculate the vectors from A to B
    
    # print ('vectors',vectors.shape)
    # print('vector from get gector ' ,vectors)
    magnitudes = np.linalg.norm(vectors, axis=0, keepdims=True) # Calculate magnitudes, keepdims for broadcasting

    # Avoid division by zero:
    magnitudes[magnitudes == 0] = 1.0 # Or another small value if you don't want to replace 0-magnitude vectors

    normalized_vectors = vectors / magnitudes  # Normalize the vectors

    return normalized_vectors

def generate_random_angles():
    """Generates a random angle from the combined range [pi/5, pi] and [-pi, -pi/5]."""
    if random.random() < 0.5:  # 50% chance for each range
        # Generate from [pi/5, pi]
        return random.uniform(math.pi / 6, math.pi/3)
    else:
        # Generate from [-pi, -pi/5]
        return random.uniform(5*math.pi/3, 11*math.pi/6)  # Negate the uniform value




def map_range(value, from_min, from_max, to_min, to_max):
    """
    Maps a value from one range to another range using linear interpolation.

    Args:
        value: The value to be mapped.
        from_min: The minimum value of the original range.
        from_max: The maximum value of the original range.
        to_min: The minimum value of the target range.
        to_max: The maximum value of the target range.

    Returns:
        The mapped value in the target range.
    """

    # Check for valid input ranges (avoid division by zero)
    if from_max - from_min == 0:
        raise ValueError("Input range cannot be zero.")
    if to_max - to_min == 0:
      raise ValueError("Target range cannot be zero.")

    # Linear transformation formula
    mapped_value = (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min
    return mapped_value




def find_matching_rows(Vector_field_surrogate, Countur_points):
    """
    Finds the indices of rows in Vector_field_surrogate that match rows in Countur_points.

    Args:
      Vector_field_surrogate: A numpy array (N1, 3).
      Countur_points: A numpy array (N2, 3).

    Returns:
      A numpy array of indices.
    """
    matching_indices = []
    for row in Countur_points:
        matches = np.where((Vector_field_surrogate[:,:] == row).all(axis=1))[0]
        matching_indices.extend(matches.tolist())
    return np.array(matching_indices)


############################################ DEPRECATED ############################################

# def Get_cube_faces(base_coordinate,upper_coordinate_x,upper_coordinate_y,upper_coordinate_z,over_factor,delta1,X_over,Y_over,Z_over):
    
#     X_over_1 = np.atleast_2d(np.linspace(int(base_coordinate*over_factor),int((base_coordinate +  delta1)*over_factor),2,dtype=int))
#     X_over_2 = np.atleast_2d(np.linspace(int((upper_coordinate_x)*over_factor),int((upper_coordinate_x + delta1)*over_factor),2,dtype=int))
#     X_over_c = np.concatenate((X_over_1,X_over_2),axis=1)
    
   
#     Y_over_1 = np.atleast_2d(np.linspace(int(base_coordinate*over_factor),int((base_coordinate +  delta1)*over_factor),2,dtype=int))
#     Y_over_2 = np.atleast_2d(np.linspace(int((upper_coordinate_y)*over_factor),int((upper_coordinate_y+delta1)*over_factor),2,dtype=int))
#     Y_over_c = np.concatenate((Y_over_1,Y_over_2),axis=1)
   
#     Z_over_1 = np.atleast_2d(np.linspace(int(base_coordinate*over_factor),int((base_coordinate +  delta1)*over_factor),2,dtype=int))
#     Z_over_2 = np.atleast_2d(np.linspace(int((upper_coordinate_z)*over_factor),int((upper_coordinate_z+delta1)*over_factor),2,dtype=int))
    
#     Z_over_c = np.concatenate((Z_over_1,Z_over_2),axis=1)
    
#     print('X_c',X_over_c)
#     print('Y_c',Y_over_c)
#     print('Z_c',Z_over_c)
#     # Len of layers of outside points 
    
#     N_layer_x = len(np.squeeze(X_over_c))
#     N_layer_y = len(np.squeeze(Y_over_c))
#     N_layer_z = len(np.squeeze(Z_over_c))
    
#     Tot_num_Count = N_layer_x*Y_over.size*Z_over.size + N_layer_y*X_over.size*Z_over.size + N_layer_z*X_over.size*Y_over.size
    
#     Countur_points= np.zeros((Tot_num_Count,3))
    
#     p = 0 # keep traks of the points
#     for x in np.squeeze(X_over_c):
#         for y in Y_over:
#             for z in Z_over:
#                 Countur_points[p,0]=int(x)
#                 Countur_points[p,1]=int(y)
#                 Countur_points[p,2]=int(z)
#                 p = p+1
#     N_points_x = p
        
#     for y in np.squeeze(Y_over_c):
#         for x in X_over:
#             for z in Z_over:
#                 Countur_points[p,0]=int(x)
#                 Countur_points[p,1]=int(y)
#                 Countur_points[p,2]=int(z)
#                 p = p+1
#     N_points_y = p     
        
#     for z in np.squeeze(Z_over_c):
#         for x in X_over:
#             for y in Y_over:
#                 Countur_points[p,0]=int(x)
#                 Countur_points[p,1]=int(y)
#                 Countur_points[p,2]=int(z)
#                 p = p+1
#     N_points_z = p         
    
#     return Countur_points,N_points_x,N_points_y,N_points_z


# def Load_countur_vectors(Vector_field,Countur_points,Init,End):

# # Split N_points_ in two parts to extract the parallel planes at the opposite facets. To draw out the single planes, halves the 
# # extracted vector
    
#     N_points_ = End-Init
    
#     Plane_1a = Countur_points[Init:Init+int(N_points_/4),:]
#     Plane_1b = Countur_points[Init+int(N_points_/4):Init+int(N_points_/2),:]
   
#     Plane_2a = Countur_points[Init+int(N_points_/2):Init+int(N_points_*3/4),:]
#     Plane_2b = Countur_points[Init+int(N_points_*3/4):End,:]
   

#     ## Origin side # Plane 1_a and Plane 1b, generate vectors
    
#     Vector_f1 = Plane_1b-Plane_1a
   
#     norms = np.linalg.norm(Vector_f1, axis=1, keepdims=True)
   
#     Scaling_factor = 0.2
    
#     Extracted_p = Plane_1a
  
#     for m in range(Extracted_p.shape[0]):
        
        
#         Vector_= Vector_f1[m,:]
#         Vector_norm = Vector_/norms[m]
        
#         # print('Vector_norm',Vector_norm)
        
#         Vector_final = Vector_norm*Scaling_factor
        
        
        
#         # Insert the vector in the relative point in the 3D matrix
#         Vector_scaled = np.zeros([1,5])
#         Vector_scaled[0,0:3] = Vector_final.reshape(1, 3)
#         # print(Vector_scaled)
#         Vector_scaled[0,3] = -1
#         Vector_scaled[0,4] = -1 # to bypass column_match function in th emain script
#         Vector_field[int(Extracted_p[m,0]) ,int(Extracted_p[m,1]), int(Extracted_p[m,2])]=Vector_scaled
        
#     ## Origin side # Plane 2_a and Plane 2b, generate vectors
    
#     Vector_f2 = Plane_2a-Plane_2b
    
#     norms = np.linalg.norm(Vector_f2, axis=1, keepdims=True)
    
    
#     Extracted_p = Plane_2b
    
    
#     for m in range(Extracted_p.shape[0]):
        
        
#         Vector_= Vector_f2[m,:]
#         Vector_norm = Vector_/norms[m]
        
#         Vector_final = Vector_norm*Scaling_factor
        
        
#         Vector_scaled = np.zeros([1,5])
#         # Insert the vector in the relative point in the 3D matrix
        
#         Vector_scaled[0,0:3] = Vector_final.reshape(1, 3)
#         # print(Vector_scaled)
#         Vector_scaled[0,3] = -1
#         Vector_scaled[0,4] = -1 # to bypass column_match function in th emain script
        
#         Vector_field[int(Extracted_p[m][0]) ,int(Extracted_p[m][1]), int(Extracted_p[m][2])]=Vector_scaled
        
        
#     return Vector_field  

#################################################################################################

  
def Field_gradient(Vector_field_tree,Vector_field_surrogate,Vector_Field,Pre_point,Temp_point,over_factor,Dl, Connections,Neuron_of_Neurites_idx,phi_temp,theta_temp):
    '''
    Vectorial form
    
    CHECKS: 1 (half remain to be checked the cas ein which multiple choices are present)

    Parameters
    ----------
    Vector_field_tree : cKDTree 
        The object stores informations about spatial position of the gradient vectors' location.
        Is used for fast access to neurites' tip neighbouring vectors.
        
    Vector_field_surrogate : Numpy array Nx3 where N is the total number of field points
        Spatial coordinates of all the gradient vectors.
        
    Vector_Field : Numpy array width x depth x hight
        Three-dimensional array which entries are the gradient vectors [0:3] the relative neurons' type [3] and the
        idx of the neuron generating the field [4]
        
    Pre_point : Numpy array Mx3 where M is the numebr of elongating branches
        Set of segments, belonging to the protruding neurites at the given iteration, defined at
        the previous iteration.
        
    Temp_point : Numpy array Mx3
        Set of temporary segments, belonging to the protruding neurites at the given iteration,
        defined at the present iteration.
       
    over_factor : Scalar
        Oversampling factor used to build the Vector_Field.
        
    Dl : Scalar
        Segment length.
        
    Connections : Numpy array of lists N_neuronsx1
        Contains the informations about which are the post-synaptic partners of the neuron in the index.
    
    Neuron_of_Neurites_idx: Numpy array Kx1
        Stores the informations about which neuron the neurites at the idx belongs to (the idx is matched with
                                                                                       the Xi,Yi,Zi and Total_length)
        
    Phi, Theta : angles in the global reference frame that describe the current processes' direction. Already masked
                

    Returns
    -------
    Final_points : Numpy array Mx3
        New set of elongation segments after the influence of the gradient field has been assessed.
        It substitutes the Temp_point outside the function in the main algorithm.
        
    Phi, Theta : Updated angles (only the growing processes) in base of the field effects.
    COMMENTS:
        
        Neurites are not affected by a field generated by a neuron already connected with their cell (from which they originate)

    '''
    # print('                     ')
    # print('                     ')
    # print('              NEW ITERATION START          ')
    # Pre_point = np.transpose(Pre_point)
    # Temp_point = np.transpose(Temp_point)
    # print('Pre_point',Pre_point)
    # print('Temp_point',Temp_point)
    # print('                     ')
    # print('                     ')
    ## Inizialize the output, Same as temp_point. So that we just need to overwrite and 
    ## if the neurite is outside the vector filed space the relative Temp_point is kept
    Final_points = Temp_point
    if np.isnan(Pre_point).any():  # Check if ANY element is NaN
        raise ValueError("NaN found in the array!")  # Or use sys.exit()
    # First the coordinates of the growing neurites must be scaled by over_factor   
    Pre_points_s =np.squeeze( Pre_point*over_factor)
    Temp_points_s = np.squeeze(Temp_point*over_factor)
  
    # Get normalized vectors of the neurites. np.atleast_2d trnasforms (3,) into (1,3)
    Neurites_vector_normilized = np.atleast_2d(Get_vectors(Pre_points_s, Temp_points_s))
    # print('Neurites_vector_normilized',Neurites_vector_normilized)
    # print('                     ')
    # print('                     ')
    # CORRECT # 
    
    # Search for the closest field point for each of the query points
    distances, indices = Vector_field_tree.query(Pre_points_s, k=1)  # Query for k-nearest
    # print('distances',distances)
    # print('indices',indices)
    # print('                     ')
    # print('                     ')
    # If tips are too distant from any point in the vector field than the distance is set at inf
    # and the idx returned is the size of Vector_field_surrogate which generate errors in accessing 
    # elements
  
    indices_isinf = False
    # Handle the case in which indicies is an INTEGER
    if np.isscalar(indices):
        
        # Infinite distance
        if np.isinf(distances):
            indices_isinf = True
            indices_post = 0 # Dummy variable, not used anyway
        else: # Not inf
        
            not_inf_indices = 0 # used for IDX_ to retrieve info about the neurites_norm_vector 
                                # and Neuron of belonging
            indices_post = indices

    else:
                
           
        # Find indices of elements that are NOT inf
        not_inf_indices = np.where(~np.isinf(distances))[0]
        
        
        # Filter Indices
        indices_post = indices[not_inf_indices]
        
        # print('Idx_filt', indices_filtred)
        # Get the actual coordinates 
        
    # print('not_inf_indices',not_inf_indices)
    # print('indices_post',indices_post)
    # print('                     ')
    # print('                     ')
        
        
    Closest_points = np.atleast_2d(Vector_field_surrogate[indices_post,:])  # Size: Nx3
    # print('Closest_points', Closest_points)
    
    # For each point find the vector influencing it
    if indices_isinf == True:
        print('indices_isinf == True')
        p_range = 0 # The for loop will NOT be executed
    
    elif isinstance(indices_post, int):
        p_range = 1
        
    else:
        p_range = len(indices_post)
    for p in range(p_range): # The iteration is for Neurites_vector_normilized, the relative influence vector are in the same index of 
                                    # Closest_points
        
        
        # print('                     ')
        # print('                     ')
        # print('                 INITIALIZATION              ',p)
        
        
        
        Influence_array = Vector_Field[int(Closest_points[p,0]),int(Closest_points[p,1]),int(Closest_points[p,2])]
        # print('Influence_array PRE ', Influence_array)
        # Index of interest to retrieve info.
        # not_inf_indices might be scalar
        
        if np.isscalar(not_inf_indices):
            IDX_ = not_inf_indices
        
        else:
            IDX_ = not_inf_indices[p]
        # Extract the idx tip's of interest after the filtering
        Neurites_  = Neurites_vector_normilized[IDX_,:] # Extract the relative neurites
        # print('Neurites_ PRE ', Neurites_)
        
        Neuron_of_belonging = Neuron_of_Neurites_idx[IDX_]
        # Should be an array of idx of the neurons' already connected.
        Linked_neuron = Connections[Neuron_of_belonging]
        # Extract the connections already established by the neurite's neuron of belonging
        
        # Assess the size of the vector
        if np.isscalar(Influence_array):  # Check for scalar zero
        
            Influence_vector = np.array([0,0,0]) # Null vector
            
        else:
           
            
            
            
            # Rule out the vectors generated by the neurons connected to the neurite's neuron
            # Because of how Connection matrix is built, the function automatically rules out 
            # the vectors generate by the same neuron
            Influence_array = remove_matching_rows(Influence_array, Linked_neuron) # Filtered matrix
            
            if Influence_array.size == 0:  # Check for empty vector
            
                Influence_vector = np.array([0,0,0]) # Null vector
            
            # Check for the number of nested vectors
            
            elif Influence_array.shape[0] == 1: # Just one vector
                
                Influence_vector = Influence_array
                
            else: # More than one vector
            
    # Rules must be established for the correct and consistent choice.
    # Multiple influence clouds acting on the same node. The growing axon will have to choose between 
    # multiple gradients in base of the following gerarchica conditions: 1) Type of neuron 2) Magnitude of the gradient
                
                # Type of neurons generating the gradient
                
                Gradient_type = Influence_array[:,3]
                
                
                
                indices_gr = np.where(Gradient_type != Neuron_of_belonging) # CONDITION TO FILTER THE AFFINE GRADIENTS
                
                # np.where returns a tuple.  If arr is 1D, you'll want the first element:
                indices_gr = indices_gr[0]
                
                if indices_gr.size == 0: # No affine gradients
                    
                    # Choose the strongest gradient
                    
                    modules = np.linalg.norm(Influence_array, axis=1)  # Calculate magnitudes of each row (vector)
                    largest_module_index = np.argmax(modules)
                    
                    Influence_vector = Influence_array[largest_module_index,:]
                            
                else: # existence of affine gradients
                
                    if indices_gr.size == 1:
                        
                        Influence_vector = Influence_array[indices_gr,:]
                        
                    else: # More than one affine gradients, choose the strongest
                        
                        # Extract the affine vectors and choose the strongest
                        Affine_gradients = Influence_array[indices_gr,:]
                        modules = np.linalg.norm(Affine_gradients, axis=1)  # Calculate magnitudes of each row (vector)
                        largest_module_index = np.argmax(modules)
                        
                        Influence_vector = Affine_gradients[largest_module_index,:]
                        
        
       
        # Sum the trajectory vector and the influence vector and retrieve the final point  
        # print('                     ')
        # print('                     ')
        # print('                 VECTOR CONSTRUCTION               ')
        # Checked
        #Reshape vector
        Influence_vector = np.squeeze(Influence_vector)
        # print('Influence_vector',Influence_vector)
        
        Vector_sum = Neurites_ + Influence_vector[0:3]
        # print('Neurites_',Neurites_)
        # print('Vector_sum',Vector_sum)
        
        # Normalize, scale with regard to the segment length Dl and find the new final point.
        Vector_sum_Norm = Vector_sum/np.linalg.norm(Vector_sum)
        # print('Vector_sum_Norm*Dl',Vector_sum_Norm*Dl)
        New_point = Pre_point[IDX_,:] + Vector_sum_Norm*Dl
        # print('Pre_point[IDX_,:]',Pre_point[IDX_,:])
        # print('New_point',New_point)
        Final_points[IDX_,:] = New_point # Rows match in size the number of elements of Xi Yi Zi 
        # print('IDX_',IDX_)
        # print('Final_points',Final_points)
        # print('Final_points[IDX_,:]',Final_points[IDX_,:])
        
        
        ### CHECK FOR THE CROSSING OF THE BORDER ###
        
        
        
        
        
        
        
        
        
        '''
        Spherical <-> cartesian frames
        
        x = r sin(theta) cos(phi)
        y = r sin(theta) sin(phi)
        z = r cose(theta)
        
        phi
        
        theta = arccos(z/sqrt(x**2 + y**2 + z**2))
        phi = sgn(y) arccos(z/sqrt(x**2 + y**2))
                    
        
        '''
        
        
        # For the given neurites update the phi theta angles, based on the normalized vector
        
        # Theta 
        
        if Vector_sum_Norm[2] > 0:
            
            theta_temp[p] = math.atan(np.sqrt(Vector_sum_Norm[0]**2+Vector_sum_Norm[1]**2)/Vector_sum_Norm[2])
            
        elif Vector_sum_Norm[2] < 0:
            
            theta_temp[p] = np.pi +  math.atan(np.sqrt(Vector_sum_Norm[0]**2+Vector_sum_Norm[1]**2)/Vector_sum_Norm[2])
            
        else:
            theta_temp[p] = np.pi/2
            
            
            
        # Phi
        if Vector_sum_Norm[0] > 0:
            
            phi_temp[p] = math.atan(Vector_sum_Norm[1]/Vector_sum_Norm[0])
            
        elif Vector_sum_Norm[0] < 0 and Vector_sum_Norm[1] >= 0:
            
            phi_temp[p] = np.pi + math.atan(Vector_sum_Norm[1]/Vector_sum_Norm[0])
         
        elif Vector_sum_Norm[0] < 0 and Vector_sum_Norm[1] < 0:
            
            phi_temp[p] = -np.pi + math.atan(Vector_sum_Norm[1]/Vector_sum_Norm[0])
        
        elif Vector_sum_Norm[0] == 0 and Vector_sum_Norm[1] > 0:
            
            phi_temp[p] = np.pi / 2
            
        elif Vector_sum_Norm[0] == 0 and Vector_sum_Norm[1] > 0:
         
            phi_temp[p] = -np.pi / 2
            
            
    return Final_points,phi_temp,theta_temp
    

def Check_synapses(X,Y,Z,A,Final_points,Connections,alpha_values,r_dendrite,Neuron_of_belonging,Ne):
    '''
    

    Parameters
    ----------
    X : N_neuronx1
        X coordinates of somata.
        
    Y : N_neuronx1
        Y coordinates of somata.
        
    Z : N_neuronx1
        Z coordinates of somata.
        
    A : N_neuronxN_neuron
        Adajency matrix.
        
    Final_points : Kx3
        In the current iteration it stores the coordinates (along columns) of the newly concatenated segments
        of the elongating branches.
        
    Connections : N_neuronsx1 np.array of lists
        Per each index it stores the informations about the post-synaptic partners of the neuron at the index.
        
    alpha_values : 3,
        Bias term for the synaptic probability of connection.[0] = exc-exc; [1] = exc-inh
        [2] = inh-exc/inh
        
    r_dendrite : N_neurons,
        Radious of the perisomatic dendritic arbourisation of the neuron at the index.    
    
    Neuron_of_belonging : Kx1
        Stores the informations about which neuron the neurites at the idx belongs to (the idx is matched with
                                                                                       the Xi,Yi,Zi and Total_length)
    Ne : Scalar
        Number of excitatory neurons

    Returns
    -------
    A : N_neuronxN_neuron
        Updated adajency matrix.
        
    Connections : N_neuronsx1 np.array of lists
        Updated Connections array.

    '''
    
    # Find per each neurite the collidin dendritic arborizations
    # Because the distance calculation and comparison are done element-wise, the final result P is an array (or list) of booleans. 
    # Each element P[i] will be:
    # True if the distance between (X, Y, Z) and the i-th point in Xj, Yj, Zj is less than r_dendrit.
    # False if the distance is greater than or equal to r_dendrit.
    # print('finalpoints shape',Final_points.shape[0])
    # Iterate across Final_points
    
    # print('                     ')
    # print('                     ')
    # print('                 SYNAPSE CHECK               ')
   
    # print('Final_points.shape[0]',Final_points.shape[0])
    for p in range(Final_points.shape[0]):
        
        # Retrieve informations:
        # print('                     ')
        # print('                     ')
        # print('                 PRELIMINAR               ')
        # print('                     ')
        # print('                     ')
        # Neurite's coordinates       
        P_coordinates = Final_points[p,:]
        # print('P_coordinates',P_coordinates)
        # Relative neuron
        Pre_syn_neuron  = Neuron_of_belonging[p]
        # print('Neuron_of_belonging[p',Neuron_of_belonging[p])
        # print('Pre_syn_neuron',Pre_syn_neuron)
        
        # Connections already set, to avoid evaluating multiple times the same connection.
        Post_set = Connections[Pre_syn_neuron]
        # print('Connections[Pre_syn_neuron]',Connections[Pre_syn_neuron])
        # print('Post_set',Post_set)
        # Post_syn idx mathces the one of the output variable in the following function 
        Q = np.sqrt(np.power(P_coordinates[0]-X,2) + np.power(P_coordinates[1]-Y,2) + np.power(P_coordinates[2] - Z,2)) < r_dendrite # =1 if condition is fullfilled.
        # print('Q',Q)
        # The idx where elements are True are the idx of the neurons in which the neurites has collided.
        Post_syn_temp = np.where(Q)[0]
        # print('Post_syn_temp',Post_syn_temp)
        # print('Post_syn_temp',Post_syn_temp)
        
        # Filter out the post_syn neurons already connected to the neurite's one       
        mask = np.isin(Post_syn_temp, Post_set)  # Create a boolean mask based on whether the elements in 
        # print('Post_set',Post_set)                                # Post_syn_temp are contained in Post_set
        # print('Post_syn_temp',Post_syn_temp)
        # print('mask',mask)
        Post_syn_ = Post_syn_temp[~mask]   # Apply the inverse mask
        # print('Post_syn_',Post_syn_)
        if Post_syn_.size == 0:
            continue
        else:
            
            for idx in Post_syn_:
                # Define the distance 
                # print('                     ')
                # print('                     ')
                # print('                 SINGLE SYN EVALUATION               ')
                # print('                     ')
                # print('                     ')
                Dist = np.sqrt(np.power(P_coordinates[0]-X[idx],2) + np.power(P_coordinates[1]-Y[idx],2) + np.power(P_coordinates[2] - Z[idx],2))
                # print('Dist',Dist)
                # Normalize the distance w.r.t. the post-syn neuron's dendritic arbour size
                alpha_dist = Dist/r_dendrite[idx]
                # print('r_dendrite[idx]',r_dendrite[idx])
                # print('alpha_dist',alpha_dist)
                # Choose the bias term
                if  Pre_syn_neuron < Ne and idx < Ne:  # Both exc
                    
                    alpha_bias = alpha_values[0]
                    
                elif  Pre_syn_neuron < Ne and idx >= Ne: # Pre-syn exc and post-syn inh
                    
                    alpha_bias = alpha_values[1]
                                    
                elif  Pre_syn_neuron >= Ne and idx < Ne: # Pre-syn inh and post-syn exc
                 
                    alpha_bias = alpha_values[2]
        
                elif  Pre_syn_neuron >= Ne and idx >= Ne: # Both inh
                 
                    alpha_bias = alpha_values[2]
                
                # print('Pre_syn_neuron',Pre_syn_neuron)
                # print('idx',idx)
                # print('alpha_bias',alpha_bias)
                alpha = alpha_dist*alpha_bias
                # print('alpha',alpha)
                
                random_number = random.uniform(0, 1)  # Generates a float between 0.0 (inclusive) and 1.0 (exclusive)
                # print('random_number',random_number)
                if  random_number < alpha: # connection established
                    # print('                     ')
                    # print('                     ')
                    # print('                 SYN FORMATION               ')
                # A: row-wise pre-syn neurons, column-wise post-syn
                    # print('A pre',A)
                    A[Pre_syn_neuron,idx] = 1
                    # print('A',A)
                # Update the Connection array
                    
                    Conn_temp = Connections[Pre_syn_neuron]
                    # print('Conn_temp post',Conn_temp)
                # Substitute the new array
                    # print('Connections pre',Connections)
                    Connections[Pre_syn_neuron]  =  np.append(Conn_temp,idx)
                    # print('Connections post',Connections)
                    
    return A, Connections

  
