"""
Created on Sat Jul 26 15:27:33 2025

@author: Admin
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



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
pitch_recsites = 7.5 # [um]  
shift = 11.25
for point in Grid:  
    
    # The x0 and y0 are the bottom left coordinates of the first rec site.
    # the 'point' coordinate is the center. A shift in coordinates is needed.
    # The 'point' coordinates are shifted along the diagonal about half the diameter.
    # Both x and y of the 'point' are shifted about sqrt(2)*radius
    
    x0 = point[0]-shift
    y0 = point[1]-shift
    rec_points = generate_grid_points(4, 4, pitch_recsites,x0,y0)
    
    Rec_sites.append(rec_points)
    
    
    
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
        
    
    
    
    
    
