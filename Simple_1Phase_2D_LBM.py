# Simple_1Phase_2D_LBM.py

# This program is intended to be a 'bridge' between my old graduate school hybrid LBM - FDM model and my shallow water lattice boltzmann model. 

# my convention: [Center, E, W, N, S, NE, NW, SE, SW] : [ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8]

# reversal vectors:
# 0 (0,0) stays 0
# 1 (1,0) flips to 2 (-1,0)
# 2 (-1,0) flips to 1 (1,0)
# 3 (0,1) flips to 4 (0,-1)
# 4 (0,-1) flips to 3 (0,1)
# 5 (1,1) flips to 8 (-1,-1)
# 6 (-1,1) flips to 7 (1,-1)
# 7 (1,-1) flips to 6 (-1,1)
# 8 (-1,-1) flips to 5 (1,1)

import numpy as np
import matplotlib.pyplot as plt


# this still needs to be changed from th older version...
w = np.array( [ 4/9 , 1/9 , 1/9 , 1/9 , 1/9 , 1/36 , 1/36 , 1/36 , 1/36 ] )# best for it to be global or later on maybe a class wide

# this still needs to be changed from the older version...
def lbm_flow_equilibration( rho , V , c , eT , w ):
    """
    For an 2D array with flattened into a 1D vector with N pixels, rho is 
    density, V is velocity (N by 2 matrix), c is the ratio of grid spacing to 
    the span of the time step, and eT, the unit vectors for lattice directions 
    (2 by 9 matrix). Takes these parameters, and declares a vector w to hold 
    directional weights. Calculates a vector snv for the square of the norm of 
    velocity and vdir representing velocity in each direction. Calculates and 
    returns a vector F_eq representing the new equbrium distriubtion function of
    the 'fluid' along the whole array. 
    """
    # snv: (400, 400) -> square of velocity magnitude
    snv = np.sum( V ** 2 , axis = 2 ) 
    
    # vdir: (400, 400, 9) -> velocity projected onto lattice directions
    vdir = (V @ eT) / c 
    
    w_aligned = w.reshape( 1 , 1 , 9 )
    rho_3d = rho[ : , : , np.newaxis ]
    snv_3d = snv[ : , : , np.newaxis ]
    
    term1 = rho_3d * w_aligned
    term2 = 1 + 3 * vdir + 4.5 * vdir ** 2 - 1.5 * ( snv_3d / ( c ** 2 ) )
    
    return term1 * term2


def Bhatnagher_Gross_Krook_init():
    """
    Sets grid parameters to define the size and number of nodes in the model.
    Sets the physical parameters, diffusivity, flow rate, relaxation rate, 
    viscosity, reynold numbers and sets the size and position of four circular 
    obstacles. Initializes the Bhatnagher_Gross_Krook method modeling trubulence
    in an imposed flow.
    """
    opp_indices = [0, 2, 1, 4, 3, 8, 7, 6, 5]
    
    run_steps =  10000# number of timesteps before stop
    nx = 400# number of horizontal pixels 
    ny = 400# number of vertical pixels 
    xmin = -6# distance to left side from the center
    xmax = 6# distance to right side from the center
    ymin = -6# distance to bottom side from the center
    ymax = 6# distance to top side from the center
    x_ = np.linspace( xmin , xmax , nx )
    y_ = np.linspace( ymin , ymax , nx )
    y , x = np.meshgrid( x_ , y_ )    
    dx = x_[ 1 ] - x_[ 0 ]  # Use the 1D x_ array
    dy = y_[ 1 ] - y_[ 0 ]  # Use the 1D y_ array
    
    dt = 0.02# timestep increment size
    
    flow_rate = 0.01# (imposed) fluid flow rate (choosen arbitrarily for stability)
    c = dx / dt# lattice parameter / speed of sound
    tau = 1.5# relaxation constant
    nu = ( tau - 0.5 ) * c * c * dt / 3.# viscosity parameter. Note that in SWLBM this will depend on slope, roughness, and depth and will be different. 
    Re = 2. / nu#Reynolds number
    
    e_ = np.array([
    [ 0 , 0 ],# center (resting state)
    [ 1 , 0 ],# +x (East)
    [ -1 , 0 ],# -x (West)
    [ 0 , 1 ],# +y (North)
    [ 0 , -1 ],# -y (South)
    [ 1 , 1],# +x+y (Northeast)
    [ -1 , 1 ],# -x+y (Northwest)
    [ 1 , -1 ],# +x-y (Southeast)
    [ -1 , -1 ]# -x-y (Southwest)
    ])
    
    eT = np.array([
    [0, 1, -1,  0,  0, 1, -1,  1, -1], # x-components
    [0, 0,  0,  1, -1, 1,  1, -1, -1]  # y-components
    ])
    
    # call the initial_conditions function right here
    # set the initial position and radius of some simple circular boundaries to test flow mechanics
    # a first circle (coefficents)
    h1 = -2.
    k1 = 4.
    r1 = .4
    # a second circle (coefficents)
    h2 = 0.
    k2 = 3.
    r2 = .6
    # a third circle (coefficents)
    h3 = 3.
    k3 = 0.
    r3 = .5
    # a forth circle (coefficents)
    h4 = -1.
    k4 = -1.
    r4 = .7
    
    # choose boolean masks for the cells inside these four circles
    circle_1 = ( x + k1 ) ** 2. + ( y + h1 ) ** 2. < r1 ** 2.
    circle_2 = ( x + k2 ) ** 2. + ( y + h2 ) ** 2. < r2 ** 2.
    circle_3 = ( x + k3 ) ** 2. + ( y + h3 ) ** 2. < r3 ** 2.
    circle_4 = ( x + k4 ) ** 2. + ( y + h4 ) ** 2. < r4 ** 2.
    # Make a single boolean mask encompassing pixels in all circles 
    circles = circle_1 | circle_2 | circle_3 | circle_4
    
    tol = 0.00001# tolerance is the wall thickness or point of bounce back
    # in this model L & R sides are in and outflow
    inflow_mask = np.abs( x - xmin ) < tol
    #outflow_mask = np.abs( x - xmax ) < tol# use later for tidal odel
    
    #inflowV = np.ones( np.count_nonzero( inflow_mask ) ) * flow_rate
    inflowR = flow_rate
    
    # Initialize a uniform moving field
    rho = np.full((ny, nx), inflowR)
    V = np.zeros((ny, nx, 2))
    
    # Set the whole "river" to move East at once
    V[:, :, 0] = flow_rate  
    V[circles, :] = 0  # Except inside the obstacles
    
    # Calculate the initial distribution based on this global flow
    forcer_flow = lbm_flow_equilibration(rho, V, c, eT, w)
    f = forcer_flow.copy()
    
    # the visualization to see what the model is doing: 

    plt.ion() # Turn on interactive mode
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    # Create initial plot objects we can update
    im1 = ax1.imshow(np.zeros((ny, nx)), origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis')

    ax1.set_title('Flow Magnitude')
    plt.colorbar(im1, ax=ax1)
    
    for tstep in range(run_steps):
        # Momentum and Velocity
        rho = np.sum(f, axis=2)
        rho = np.clip(rho, 0.1, 10.0)
        
        rho_v = f @ eT.T * c
        V = rho_v / rho[:, :, np.newaxis]
        
        # Relaxation (Collision)
        feq = lbm_flow_equilibration(rho, V, c, eT, w)
        omega = 1.0 / tau
        rhs = f - omega * (f - feq)
        
        # Streaming (The Movement)
        for i in range(1, 9): 
            shift_x, shift_y = e_[i]
            f[:, :, i] = np.roll(rhs[:, :, i], shift=(shift_y, shift_x), axis=(0, 1))
            
            # Zero out the 'wraparound' edges
            if shift_x > 0: f[:, 0, i] = 0  
            if shift_x < 0: f[:, -1, i] = 0 
            if shift_y > 0: f[0, :, i] = 0  
            if shift_y < 0: f[-1, :, i] = 0

        # Outflow: Open the drain
        out_vecs = [1, 5, 7] 
        for v in out_vecs:
            f[:, -1, v] = f[:, -2, v]
        
        # Inflow: Re-apply the pump
        f[inflow_mask, :] = forcer_flow[inflow_mask, :]

        f_circles_previous = f[circles, :].copy()
        f[circles, :] = f_circles_previous[:, opp_indices]

        if tstep % 10 == 0:
            mag = np.sqrt(V[:, :, 0]**2 + V[:, :, 1]**2)
            im1.set_data(mag)
            plt.pause(0.001)
            print(f"Time Step: {tstep}")

Bhatnagher_Gross_Krook_init()

