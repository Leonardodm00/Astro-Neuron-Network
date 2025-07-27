import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def generate_grid_points(n_rows, n_cols, pitch,x0,y0):
    """
    Generates the 2D coordinates for a grid of electrodes.

    Args:
        n_rows (int): Number of rows in the electrode grid.
        n_cols (int): Number of columns in the electrode grid.
        pitch_micrometers (float): The distance between adjacent electrodes
                                   in micrometers.
        x0,y0 = bottom left end coordinates.
    Returns:
        numpy.ndarray: A 2D array where each row represents the (x, y)
                       coordinates of an electrode point, in micrometers.
    """
    points = []
    # Pitch is used directly in micrometers as requested
    

    for r in range(n_rows):
        for c in range(n_cols):
            x = x0 + c * pitch
            y = y0 + r * pitch
            points.append((x, y))

    return np.array(points)



def Electrode_recording(MEA_dict,Neuron_group,State_Monitor,electrode_dist):
    
    # Generate the KDTree form the neuronal position data
    x_pos = np.array(Neuron_group[:].x)
    y_pos = np.array(Neuron_group[:].y)
    
    pos = np.column_stack((x_pos, y_pos))

    # Generate the KDTree
    Neuron_positions = KDTree(pos)
    
    
    Electrode_recordings = {}
    
    for key in MEA_dict.keys():
        
        rec_sites = MEA_dict[key]
        
        Electrode_rec = Electrode_trace(rec_sites,Neuron_group,Neuron_positions,State_Monitor,electrode_dist) 

        Electrode_recordings[key] = Electrode_rec
        
        
    return Electrode_recordings
    
    
    
    
    
    
    



def Electrode_trace(rec_sites,Neuron_group,Neuron_positions,State_Monitor,electrode_dist):

    '''
    Dipole approximation.
    
    Two contributes for background noise: 1) distant neurons, 2) White noise
    
    For each recording site the sum of all the contributing neurons is taken and 
    for the whole electrode the mean across the recording sites.
    
    electrode_dist = max senstivity distance of the electrode
    Neuron_psoitions = sklearn.neighbors.NearestNeighbors object
    
    '''
    
    # For each recording site extract the recorded neurons

    
    Site_voltages = {}
    s = 0
    for site in rec_sites:
        # For each recording site extract the recorded neurons
        NN_idx,NN_dist = Neuron_positions.radius_neighbors(site, radius=electrode_dist, return_distance=True)
       
        Voltages = State_Monitor[NN_idx].V/mV * 1/np.sqrt( (site[0] - Neuron_group[NN_idx].x)**2 + (site[1] - Neuron_group[NN_idx].y)**2)
        
        # Sum column-wise
        # ....
        # Voltages_sum
        
        # Save
        Site_voltages[s] = Voltages_sum
        s = s+1
                                          

    # --- MEAN ---
    
    # Extract traces
    rec_list = [Site_voltages[key] for key in Site_voltages.keys()]   
    rec_list_ = np.vstack(rec_list)
    
    # Take the mean
    Electrode_trace = np.mean(rec_list_, axis=0)
    
    return Electrode_trace
                 
                                        
        
        
        
    
    
        
        
    













def Get_12grid(pitch):
    
    '''
    Generates a set of coordinates for the 12 electrode MEA configuration
    
      x x
    x x x x
    x x x x
      x x
 
    '''
    
    
    x0 = 0
    y0 = 0
    pitch = 300 # [um]
    
    Grid_raw = generate_grid_points(4, 4, pitch,x0,y0)
    
    
    # The configuration is 12 electrodes. For this reason we will
    # discard the following points: 0,3,12,15
    
    Idx_remove = np.array([0,3,12,15])
    
    Grid = np.delete(Grid_raw, Idx_remove, axis=0)
    
    return Grid
    
    
    
    
pitch = 300 #[um] 
radius = 15 #[um]

Grid = Get_12grid(pitch)
  



# ------------- GRID PLOT -------------
fig, ax = plt.subplots() # This creates both a figure and an axes for you

normalized_gold_rgb = (1.0, 215 / 255, 0.0)

# 2. Iterate through each point in the Grid and add a circle for each
for point in Grid:
    # point will be an array like [x, y]
    circle = plt.Circle((point[0], point[1]), radius, color=normalized_gold_rgb, fill=True)
    ax.add_patch(circle) # Add each circle patch to the axes

# 3. Set aspect ratio to equal so circles don't look like ellipses
ax.set_aspect('equal', adjustable='box')
min_x = np.min(Grid[:, 0]) - radius * 1.2 # Add some buffer
max_x = np.max(Grid[:, 0]) + radius * 1.2
min_y = np.min(Grid[:, 1]) - radius * 1.2
max_y = np.max(Grid[:, 1]) + radius * 1.2

ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)

plt.xlabel("[um]")
plt.ylabel("[um]")
plt.show()  
    
    
   #%% 
    
# ------------- REC SITES -------------    
    
Rec_sites = []
MEA_dict = {}
pitch_recsites = 7.5 # [um]  
shift = 11.25
el = 0
for point in Grid:  
    
    # The x0 and y0 are the bottom left coordinates of the first rec site.
    # the 'point' coordinate is the center. A shift in coordinates is needed.
    # The 'point' coordinates are shifted along the diagonal about half the diameter.
    # Both x and y of the 'point' are shifted about sqrt(2)*radius
    
    x0 = point[0]-shift
    y0 = point[1]-shift
    rec_points = generate_grid_points(4, 4, pitch_recsites,x0,y0)
    MEA_dict[el] = np.array(rec_points)
    
    Rec_sites.append(rec_points)
    el = el+1
    
    
Rec_sites = np.vstack(Rec_sites)  



 
# ------------- GRID PLOT complete -------------
fig, ax = plt.subplots() # This creates both a figure and an axes for you

normalized_gold_rgb = (1.0, 215 / 255, 0.0)

# 2. Iterate through each point in the Grid and add a circle for each
for point in Grid:
    # point will be an array like [x, y]
    circle = plt.Circle((point[0], point[1]), radius, color=normalized_gold_rgb, fill=True)
    ax.add_patch(circle) # Add each circle patch to the axes

ax.scatter(Rec_sites[:,0],Rec_sites[:,1],c='k', s=5, label='Recording Sites', alpha=0.5)

# 3. Set aspect ratio to equal so circles don't look like ellipses
ax.set_aspect('equal', adjustable='box')
min_x = np.min(Grid[:, 0]) - radius * 1.2 # Add some buffer
max_x = np.max(Grid[:, 0]) + radius * 1.2
min_y = np.min(Grid[:, 1]) - radius * 1.2
max_y = np.max(Grid[:, 1]) + radius * 1.2

ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)

plt.xlabel("[um]")
plt.ylabel("[um]")
plt.show()  
        
    
    
    
    
    
    
    
    #%%
    
    # Functionf
    
def Recording_sites(pitch_recsites,shift):
    MEA_dict = {}
    
    el = 0
    for point in Grid:  
        
        # The x0 and y0 are the bottom left coordinates of the first rec site.
        # the 'point' coordinate is the center. A shift in coordinates is needed.
        # The 'point' coordinates are shifted along the diagonal about half the diameter.
        # Both x and y of the 'point' are shifted about sqrt(2)*radius
        
        x0 = point[0]-shift
        y0 = point[1]-shift
        rec_points = generate_grid_points(4, 4, pitch_recsites,x0,y0)
        MEA_dict[el] = np.array(rec_points)
        
        Rec_sites.append(rec_points)
        el = el+1
        
        
    return MEA_dict
        
    
    
    
    
    
    
    
    
    
