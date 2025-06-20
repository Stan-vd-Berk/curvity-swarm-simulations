#### This class has the various green functions used in the simulation ####
#### Different "green functions" allow for a flexible definiton of different particle-praticle interaction rules
### This can include physical types of interactions (repulsive/attractive potential, hydrodynamic interaction)
### It can also include "active matter" parameters (speed and alignment strength)
### It can also include agent based interactions (such as exchange of internal states, say speed or alignmnet strength)
### The functions allow heterogeneous population (various velocities radii etc)
### The file is structured as a set of NUMBAed functions with njit (starting with an underscore i.e _grHeteroPop)
### And a container class (greens) that calls the different functions.
### The data structure is a matrix, ps:  (number of paramteres)X(number of particles) and a class, eco: containing the indices of the params.
### The green function is a partial time step implementation for the 5th order runge kuta in timePropogation.py
### It takes the current state of the system "sp", and returns the change into "sP", which then goes to the propogator

### Updated 20220611 by Matan Clean-up
### Updated 20220527 by Matan to allow periodic boundary condition
import numba
from numba import jit, njit, int64, float32
import numpy as np
#from numba import autojit, prange


################################################# Heterogeneous Interacting Population ########################################################################
@numba.njit(parallel=True)
def _grHeteroInteractingPop(eco,sp,sP):
	'''
	input:
	eco: ecoSystem instance
	sp: ecosystem matrix float32(paramsXN) of the current state of the system
	sP: the increment matrix float32(paramsXN) based on interactions rules.
	'''
	# receives eco: system c
	### This is the main function to define the dynamics to propogate a heterogenenous population with alignment interaction, steric repulsion inherent speed.
	### Here x,y, nx, ny are affected by forces etc.
	### Size, v0, alignment and such are affected by "communication"
	### The time change for each should be slightly different 
	### DIfferent elements should be referenced through their respective position in sp table (say eco.sp[eco.xC,:]+=...)
	
	r2Tol = 1e-16;
	NN = eco.N;

	# Assign shorter names for positions and orienations arrays:
	
	sx = sp[eco.xC,:]; # X coordinates
	sy = sp[eco.yC,:]; # Y coordinates
	
	snx = sp[eco.nxC,:]; # orientation x component
	sny = sp[eco.nyC,:]; # orientation y component

	
	wS = sp[eco.wSC,:];
	v0 = sp[eco.v0C,:];
	wA = sp[eco.wAC,:];
	rSteric = sp[eco.rStericC,:];


	## The following fetch the parameters of the first particle thus treating the population as homogeneous.
	rStericCentral = rSteric[0] * 50; # Make a repulsive potential 20 times larger than particles

	## Calculate the squares for later reference
	rStericCentral2 = np.power(rStericCentral,2)

	
	for i in numba.prange(NN):
		for j in range(NN):
			if i != j:
				dx = sx[i] - sx[j]
				dy = sy[i] - sy[j]
				r2 = np.power(dx,2) + np.power(dy,2)
				if r2 > r2Tol:

					fSteric = 0
					rStericSum = rSteric[i]+rSteric[j];
					rSteric2 = np.power(rStericSum,2)
					if r2<rSteric2:
						r = np.sqrt(r2)
						irSteric = rStericSum/r
						fSteric = -wS[j]*(1-irSteric)
#                       pdb.set_trace()
						sP[eco.xC,i] += dx*fSteric 
						sP[eco.yC,i] += dy*fSteric

		fCentral = 0
		fAllign = 0
		rCentral2= np.power(sx[i],2)+np.power(sy[i],2) #Assuming potential is centered at origin
		
		if rCentral2 < rStericCentral2:
			rCentral = np.sqrt(rCentral2)
			irCentral = rStericCentral/rCentral
			fCentral = -wS[i]*(1-irCentral)
			fAllign = wA[i]*(snx[i]*sy[i]-sny[i]*sx[i]) #n cross force
 
            #central force is located at the origin (otherwise add sx[i]-sxCentral etc.)
			sP[eco.nxC,i] +=  - sny[i]*fAllign
			sP[eco.nyC,i] +=  + snx[i]*fAllign#(sny[i] - snx[i]*fAllign)*normFAllign - sny[i]
        
		
		sP[eco.xC,i] += sx[i]*fCentral + v0[i]*snx[i] #currently spin is not normalized
		sP[eco.yC,i] += sy[i]*fCentral + v0[i]*sny[i]
"""def _MIPSHeteroPop(eco, sp, sP, boxSize):

"""
################################################# Heterogeneous Interacting Population in a Periodic Box #######################################################

@njit
def compute_minimum_distance(pos_i, pos_j, box_size):
    r_ij = pos_j - pos_i
    r_ij = r_ij - box_size * np.round(r_ij / box_size)
    return r_ij

@njit
def compute_repulsive_force(pos_i, pos_j, radius_i, radius_j, stiffness, box_size):
    r_ij = compute_minimum_distance(pos_i, pos_j, box_size)
    dist = np.sqrt(np.sum(r_ij ** 2))

    if dist < 1e-10:
        r_ij = np.array([1e-5, 1e-5], dtype=np.float32)  # Make sure it's float32
        dist = np.sqrt(np.sum(r_ij ** 2))

    r_hat = r_ij / dist
    sum_radii = radius_i + radius_j

    if dist < sum_radii:
        force_magnitude = stiffness * (sum_radii - dist)
        return -force_magnitude * r_hat

    return np.zeros(2, dtype=np.float32)  # Ensure returning float32

@njit
def create_cell_list(positions, box_size, cell_size, n_particles):
    n_cells = int(np.floor(box_size / cell_size))
    head = np.ones((n_cells, n_cells), dtype=int64) * -1
    list_next = np.ones(n_particles, dtype=int64) * -1

    for i in range(n_particles):
        cell_x = int(positions[i, 0] / cell_size)
        cell_y = int(positions[i, 1] / cell_size)

        # Clamp to box boundaries (if needed)
        cell_x = min(cell_x, n_cells - 1)
        cell_y = min(cell_y, n_cells - 1)

        list_next[i] = head[cell_x, cell_y]
        head[cell_x, cell_y] = i

    return head, list_next, n_cells

@njit(parallel=True)
def compute_all_forces(positions, radii, stiffness, box_size):
    n_particles = positions.shape[0]
    forces = np.zeros((n_particles, 2), dtype=np.float32)

    # Use cell list for efficiency
    max_radius = np.max(radii)
    cell_size = 2.0 * max_radius
    head, list_next, n_cells = create_cell_list(positions, box_size, cell_size, n_particles)

    for i in prange(n_particles):
        pos_i = positions[i]
        radius_i = radii[i]

        # Determine which cell the particle is in
        cell_x = int(pos_i[0] / cell_size)
        cell_y = int(pos_i[1] / cell_size)

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_x = (cell_x + dx) % n_cells
                neighbor_y = (cell_y + dy) % n_cells

                j = head[neighbor_x, neighbor_y]
                while j != -1:
                    if j != i:
                        pos_j = positions[j]
                        radius_j = radii[j]
                        f_ij = compute_repulsive_force(pos_i, pos_j, radius_i, radius_j, stiffness, box_size)
                        forces[i] += f_ij
                    j = list_next[j]

    return forces

@njit(parallel=True)
def _grInteractionHeteroPopPeriodicOptimized(eco, sp, sP, boxSize):
    NN = eco.N
    sP = np.zeros(np.shape(sP), dtype=np.float32)

    positions = np.zeros((NN, 2), dtype=np.float32)
    positions[:, 0] = sp[eco.xC, :]
    positions[:, 1] = sp[eco.yC, :]

    rSteric = sp[eco.rStericC, :].astype(np.float32)  # Ensure it's float32
    wS = sp[eco.wSC, :].astype(np.float32)  # Ensure it's float32
    avg_wS = np.mean(wS)

    forces = compute_all_forces(positions, rSteric, avg_wS, boxSize)

    sP[eco.xC, :] = forces[:, 0]
    sP[eco.yC, :] = forces[:, 1]

    return sP
@numba.njit(parallel=True)
def _grInteractingHeteroPopPeriodic(eco,sp,sP,boxSize):
#Aligning particles 
### This is the main function to define the dynamics to propogate a heterogenenous population with alignment interaction, steric repulsion inherent speed.
### Here x,y, nx, ny are affected by forces etc.
### Size, v0, alignment and such are affected by "communication"
### The time change for each should be slightly different 
### DIfferent elements should be referenced through their respective position in sp table (say eco.sp[eco.xC,:]+=...)

	''' green function for only steric interactions with a heterogenous population'''
	rTol = 1E-8; # (original)
	#rTol = 0.000633; ## Consistent with periodic boundary condition
	r2Tol = np.power(rTol,2);

	NN = eco.N;

	#periodicReg = 0.0015; #regularization parameter for smooth periodic boundary condition
	#periodicReg2 = np.power(periodicReg,2)
	# Assign shorter names for positions and orienations arrays:
	
	sx = sp[eco.xC,:]; # X coordinates
	sy = sp[eco.yC,:]; # Y coordinates
	
	snx = sp[eco.nxC,:]; # orientation x component
	sny = sp[eco.nyC,:]; # orientation y component

	
	wS = sp[eco.wSC,:];
	v0 = sp[eco.v0C,:];
	wA = sp[eco.wAC,:];
	rSteric = sp[eco.rStericC,:];

	for i in numba.prange(NN):
		#normalie mobility to mobility of first particle
		mobility = 1/sp[eco.rStericC,i]
		for j in range(NN):
			if i != j:
				dx = sx[i] - sx[j]
				dy = sy[i] - sy[j]
				
				#find vectorial minimal separation on a torus
				dxA = np.abs(dx)
				dyA = np.abs(dy)
			
				
				if dxA > boxSize/2:
					dx = np.sign(dx)*(dxA-boxSize)
						
				if dyA > boxSize/2:
					dy = np.sign(dy)*(dyA-boxSize)
					
					
				r2 = np.power(dx,2) + np.power(dy,2)
				
				if r2 > r2Tol:

					fSteric = 0
					rStericSum = rSteric[i]+rSteric[j];
					rSteric2 = np.power(rStericSum,2)
					if r2<rSteric2:
						r = np.sqrt(r2)
						irSteric = rStericSum/r
						fSteric = -wS[j]*(1-irSteric)
#                       pdb.set_trace()
						sP[eco.xC,i] += dx*fSteric*mobility
						sP[eco.yC,i] += dy*fSteric*mobility

 
        #central force is located at the origin (otherwise add sx[i]-sxCentral etc.)
		#aligning force (Fxn) is the cross product of the total sum of forces on the particle and orientation 
		fAllign = wA[i]*(snx[i]*sP[eco.yC,i]-sny[i]*sP[eco.xC,i]) #n cross force
		sP[eco.nxC,i] +=  - sny[i]*fAllign
		sP[eco.nyC,i] +=  + snx[i]*fAllign#(sny[i] - snx[i]*fAllign)*normFAllign - sny[i]
			

		sP[eco.xC,i] += v0[i]*snx[i] #currently spin is not normalized
		sP[eco.yC,i] += v0[i]*sny[i]		
		
################################################# Heterogeneous only passive interacting in a Periodic Box #######################################################	
@numba.njit(parallel=True)
def _grNonInteractingHeteroWithPassivePopPeriodic(eco,sp,sP,boxSize):
#Aligning particles 
### This is the main function to define the dynamics to propogate a heterogenenous population with alignment interaction, steric repulsion inherent speed.
### Here x,y, nx, ny are affected by forces etc.
### Size, v0, alignment and such are affected by "communication"
### The time change for each should be slightly different 
### DIfferent elements should be referenced through their respective position in sp table (say eco.sp[eco.xC,:]+=...)

	''' green function for only steric interactions with a heterogenous population'''
	rTol = 1E-8; # (original)
	#rTol = 0.000633; ## Consistent with periodic boundary condition
	r2Tol = np.power(rTol,2);

	NN = eco.N;

	#periodicReg = 0.0015; #regularization parameter for smooth periodic boundary condition
	#periodicReg2 = np.power(periodicReg,2)
	# Assign shorter names for positions and orienations arrays:
	
	sx = sp[eco.xC,:]; # X coordinates
	sy = sp[eco.yC,:]; # Y coordinates
	
	snx = sp[eco.nxC,:]; # orientation x component
	sny = sp[eco.nyC,:]; # orientation y component

	
	wS = sp[eco.wSC,:];
	v0 = sp[eco.v0C,:];
	wA = sp[eco.wAC,:];
	rSteric = sp[eco.rStericC,:];

	for i in numba.prange(NN):
		for j in range(NN):
			if ((i != j) and (v0[i]*v0[j]==0)):
				dx = sx[i] - sx[j]
				dy = sy[i] - sy[j]
				
				#find vectorial minimal separation on a torus
				dxA = np.abs(dx)
				dyA = np.abs(dy)
			
				
				if dxA > boxSize/2:
					dx = np.sign(dx)*(dxA-boxSize)
						
				if dyA > boxSize/2:
					dy = np.sign(dy)*(dyA-boxSize)
					
					
				r2 = np.power(dx,2) + np.power(dy,2)
				
				if r2 > r2Tol:

					fSteric = 0
					rStericSum = rSteric[i]+rSteric[j];
					rSteric2 = np.power(rStericSum,2)
					if r2<rSteric2:
						r = np.sqrt(r2)
						irSteric = rStericSum/r
						fSteric = -wS[j]*(1-irSteric)
#                       pdb.set_trace()
						sP[eco.xC,i] += dx*fSteric 
						sP[eco.yC,i] += dy*fSteric

 
        #central force is located at the origin (otherwise add sx[i]-sxCentral etc.)
		#aligning force (Fxn) is the cross product of the total sum of forces on the particle and orientation 
		fAllign = wA[i]*(snx[i]*sP[eco.yC,i]-sny[i]*sP[eco.xC,i]) #n cross force
		sP[eco.nxC,i] +=  - sny[i]*fAllign
		sP[eco.nyC,i] +=  + snx[i]*fAllign#(sny[i] - snx[i]*fAllign)*normFAllign - sny[i]
			

		sP[eco.xC,i] += v0[i]*snx[i] #currently spin is not normalized
		sP[eco.yC,i] += v0[i]*sny[i]		

		
################################################# Heterogeneous Interacrting with a Central Potential #######################################################	
@numba.njit(parallel=True)
def _grInteractingHeteroPopCentralPot(eco,sp,sP):
#Aligning particles 
### This is the main function to define the dynamics to propogate a heterogenenous population with alignment interaction, steric repulsion inherent speed.
### Here x,y, nx, ny are affected by forces etc.
### Size, v0, alignment and such are affected by "communication"
### The time change for each should be slightly different 
### DIfferent elements should be referenced through their respective position in sp table (say eco.sp[eco.xC,:]+=...)


	''' green function for only steric interactions with a heterogeneous population (using the first particle) and a central potential'''
	
	r2Tol = 1e-16;
	NN = eco.N;


	# Assign shorter names for positions and orienations arrays:
	
	sx = sp[eco.xC,:]; # X coordinates
	sy = sp[eco.yC,:]; # Y coordinates
	
	snx = sp[eco.nxC,:]; # orientation x component
	sny = sp[eco.nyC,:]; # orientation y component

	
	wS = sp[eco.wSC,:];
	v0 = sp[eco.v0C,:];
	wA = sp[eco.wAC,:];
	rSteric = sp[eco.rStericC,:];


	## The following fetch the parameters of the first particle thus treating the population as homogeneous.
	rStericCentral = rSteric[0] * 50; # Make a repulsive potential 20 times larger than particles

	## Calculate the squares for later reference
	rStericCentral2 = np.power(rStericCentral,2)

	
	for i in numba.prange(NN):
		for j in range(NN):
			if i != j:
				dx = sx[i] - sx[j]
				dy = sy[i] - sy[j]
				r2 = np.power(dx,2) + np.power(dy,2)
				if r2 > r2Tol:

					fSteric = 0
					rStericSum = rSteric[i]+rSteric[j];
					rSteric2 = np.power(rStericSum,2)
					if r2<rSteric2:
						r = np.sqrt(r2)
						irSteric = rStericSum/r
						fSteric = -wS[j]*(1-irSteric)
#                       pdb.set_trace()
						sP[eco.xC,i] += dx*fSteric 
						sP[eco.yC,i] += dy*fSteric

		fCentral = 0
		fAllign = 0
		rCentral2= np.power(sx[i],2)+np.power(sy[i],2) #Assuming potential is centered at origin
		
		if rCentral2 < rStericCentral2:
			rCentral = np.sqrt(rCentral2)
			irCentral = rStericCentral/rCentral
			fCentral = -wS[i]*(1-irCentral)
			sP[eco.xC,i] += sx[i]*fCentral #currently spin is not normalized
			sP[eco.yC,i] += sy[i]*fCentral
 
            #central force is located at the origin (otherwise add sx[i]-sxCentral etc.)
		#aligning force (Fxn) is the cross product of the total sum of forces on the particle and orientation 
		fAllign = wA[i]*(snx[i]*sP[eco.yC,i]-sny[i]*sP[eco.xC,i]) #n cross force
		sP[eco.nxC,i] +=  - sny[i]*fAllign
		sP[eco.nyC,i] +=  + snx[i]*fAllign#(sny[i] - snx[i]*fAllign)*normFAllign - sny[i]
			

		sP[eco.xC,i] += v0[i]*snx[i] #currently spin is not normalized
		sP[eco.yC,i] += v0[i]*sny[i]	


@numba.njit(parallel=True)
def _grInteractingHeteroPopCircularWall(eco, sp, sP, boxSize):
    '''
    Green's function with:
    - Steric interaction
    - Circular wall centered at (boxSize/2, boxSize/2)
    - Alignment + self propulsion
    - Parallel-safe version
    '''

    r2Tol = 1e-16
    NN = eco.N

    sx = sp[eco.xC, :]
    sy = sp[eco.yC, :]
    snx = sp[eco.nxC, :]
    sny = sp[eco.nyC, :]
    wS = sp[eco.wSC, :]
    v0 = sp[eco.v0C, :]
    wA = sp[eco.wAC, :]
    rSteric = sp[eco.rStericC, :]

    cx = boxSize / 2
    cy = boxSize / 2
    radius = boxSize / 2
    maxRadius2 = (radius * 0.99) ** 2

    for i in numba.prange(NN): 
        for j in range(NN):
            if i != j:
                dx = sx[i] - sx[j]
                dy = sy[i] - sy[j]
                r2 = dx * dx + dy * dy
                if r2 > r2Tol:
                    rStericSum = rSteric[i] + rSteric[j]
                    rSteric2 = rStericSum * rStericSum
                    if r2 < rSteric2:
                        r = np.sqrt(r2)
                        irSteric = rStericSum / r
                        fSteric = -wS[j] * (1 - irSteric)
                        sP[eco.xC, i] += dx * fSteric
                        sP[eco.yC, i] += dy * fSteric
        dxC = sx[i] - cx
        dyC = sy[i] - cy
        rWall2 = dxC * dxC + dyC * dyC
        if rWall2 > maxRadius2:
            rWall = max(np.sqrt(rWall2), 1e-6)
            scale = (radius * 0.99) / rWall
            fx = (dxC - dxC * scale) * wS[i] * 10.0
            fy = (dyC - dyC * scale) * wS[i] * 10.0
            sP[eco.xC, i] -= fx
            sP[eco.yC, i] -= fy

        # Alignment
        fAlign = wA[i] * (snx[i] * sP[eco.yC, i] - sny[i] * sP[eco.xC, i])
        sP[eco.nxC, i] += -sny[i] * fAlign
        sP[eco.nyC, i] += +snx[i] * fAlign

        # Self-propulsion
        sP[eco.xC, i] += v0[i] * snx[i]
        sP[eco.yC, i] += v0[i] * sny[i]

        """# Normalize orientation (optional)
        norm_sq = snx[i] * snx[i] + sny[i] * sny[i]
        if norm_sq < 1e-12:
            norm_sq = 1.0
        inv_norm = 1.0 / np.sqrt(norm_sq)
        snx[i] *= inv_norm
        sny[i] *= inv_norm"""

    return sP



@numba.njit(parallel=True)
def _grInteractingHeteroPopRectangularWall(eco, sp, sP, boxSize):
    '''
    Green's function with:
    - Steric particle interaction
    - Rectangular soft walls at boundaries [0, boxSize]
    - Alignment and self-propulsion
    '''

    r2Tol = 1e-16
    NN = eco.N

    sx = sp[eco.xC, :]
    sy = sp[eco.yC, :]
    snx = sp[eco.nxC, :]
    sny = sp[eco.nyC, :]
    wS = sp[eco.wSC, :]
    v0 = sp[eco.v0C, :]
    wA = sp[eco.wAC, :]
    rSteric = sp[eco.rStericC, :]

    for i in numba.prange(NN):
        # Inter-particle steric repulsion
        for j in range(NN):
            if i != j:
                dx = sx[i] - sx[j]
                dy = sy[i] - sy[j]
                r2 = dx * dx + dy * dy
                if r2 > r2Tol:
                    rStericSum = rSteric[i] + rSteric[j]
                    rSteric2 = rStericSum * rStericSum
                    if r2 < rSteric2:
                        r = np.sqrt(r2)
                        irSteric = rStericSum / r
                        fSteric = -wS[j] * (1 - irSteric)
                        sP[eco.xC, i] += dx * fSteric
                        sP[eco.yC, i] += dy * fSteric

        # Rectangular wall confinement force (soft repulsion)
        margin = rSteric[i] * 1.2
        wall_strength = wS[i] * 10.0

        if sx[i] < margin:
            sP[eco.xC, i] += wall_strength * (margin - sx[i])
        elif sx[i] > boxSize - margin:
            sP[eco.xC, i] -= wall_strength * (sx[i] - (boxSize - margin))

        if sy[i] < margin:
            sP[eco.yC, i] += wall_strength * (margin - sy[i])
        elif sy[i] > boxSize - margin:
            sP[eco.yC, i] -= wall_strength * (sy[i] - (boxSize - margin))

        # Alignment torque
        fAlign = wA[i] * (snx[i] * sP[eco.yC, i] - sny[i] * sP[eco.xC, i])
        sP[eco.nxC, i] += -sny[i] * fAlign
        sP[eco.nyC, i] += +snx[i] * fAlign

        # Self propulsion
        sP[eco.xC, i] += v0[i] * snx[i]
        sP[eco.yC, i] += v0[i] * sny[i]

    return sP



			
################################################# Heterogeneous NonInteracrting with a Central Potential #######################################################	
@numba.njit(parallel=True)		
def _grNonInteractingHeteroPopCentralPot(eco,sp,sP):
### This is the main function to define the dynamics to propogate a heterogenenous population with alignment interaction, steric repulsion inherent speed.
### Here x,y, nx, ny are affected by forces etc.
### Size, v0, alignment and such are affected by "communication"
### The time change for each should be slightly different 
### DIfferent elements should be referenced through their respective position in sp table (say eco.sp[eco.xC,:]+=...)

### -------------->>>>>>>>In fact I should add stuff like the following: <<<<<<<<------------------------------- 20200520 Matan
## wS = sp[eco.wSC,i] and such at begininng of each loop
## v0 = sp[eco.v0C,i] and such at begininng of each loop
## rSteric = sp[eco.rStericC,i] and such at begininng of each loop
## rSteric2 = (rStericI+rStericJ)**2
## Need to declare the external, global sp, to mimic the capilaized arrays such as sY sX etc.

	''' green function for only steric interactions with a homogeneous population (using the first particle)'''
	
	r2Tol = 1e-16;
	NN = eco.N;


	# Assign shorter names for positions and orienations arrays:
	
	sx = sp[eco.xC,:]; # X coordinates
	sy = sp[eco.yC,:]; # Y coordinates
	
	snx = sp[eco.nxC,:]; # orientation x component
	sny = sp[eco.nyC,:]; # orientation y component

	
	wS = sp[eco.wSC,:];
	v0 = sp[eco.v0C,:];
	wA = sp[eco.wAC,:];
	rSteric = sp[eco.rStericC,:];


	## The following fetch the parameters of the first particle thus treating the population as homogeneous.
	rStericCentral = rSteric[0] * 50; # Make a repulsive potential 20 times larger than particles

	## Calculate the squares for later reference
	rStericCentral2 = np.power(rStericCentral,2)

	
	for i in numba.prange(NN):
		fCentral = 0
		fAllign = 0
		rCentral2= np.power(sx[i],2)+np.power(sy[i],2) #Assuming potential is centered at origin
		
		if rCentral2 < rStericCentral2:
			rCentral = np.sqrt(rCentral2)
			irCentral = rStericCentral/rCentral
			fCentral = -wS[i]*(1-irCentral)
			fAllign = wA[i]*(snx[i]*sy[i]-sny[i]*sx[i]) #n cross force
 
            #central force is located at the origin (otherwise add sx[i]-sxCentral etc.)
			sP[eco.nxC,i] +=  - sny[i]*fAllign
			sP[eco.nyC,i] +=  + snx[i]*fAllign#(sny[i] - snx[i]*fAllign)*normFAllign - sny[i]
        
		
		sP[eco.xC,i] += sx[i]*fCentral + v0[i]*snx[i] #currently spin is not normalized
		sP[eco.yC,i] += sy[i]*fCentral + v0[i]*sny[i]
	
################################################# Heterogeneous NonInteracrting with a Constant Force #######################################################	
@numba.njit(parallel=True)
def _grInteractingHeteroPopConstantForce(eco,sp,sP):
#Aligning particles 
### This is the main function to define the dynamics to propogate a heterogenenous population with alignment interaction, steric repulsion inherent speed.
### Here x,y, nx, ny are affected by forces etc.
### Size, v0, alignment and such are affected by "communication"
### The time change for each should be slightly different 
### DIfferent elements should be referenced through their respective position in sp table (say eco.sp[eco.xC,:]+=...)


	''' green function for only steric interactions with a heterogeneous population (using the first particle) and a constant force (like down a slope along y)'''
	
	r2Tol = 1e-16;
	NN = eco.N;


	# Assign shorter names for positions and orienations arrays:
	
	sx = sp[eco.xC,:]; # X coordinates
	sy = sp[eco.yC,:]; # Y coordinates
	
	snx = sp[eco.nxC,:]; # orientation x component
	sny = sp[eco.nyC,:]; # orientation y component

	
	wS = sp[eco.wSC,:];
	v0 = sp[eco.v0C,:];
	wA = sp[eco.wAC,:];
	rSteric = sp[eco.rStericC,:];

	
	for i in numba.prange(NN):
		for j in range(NN):
			if i != j:
				dx = sx[i] - sx[j]
				dy = sy[i] - sy[j]
				r2 = np.power(dx,2) + np.power(dy,2)
				if r2 > r2Tol:

					fSteric = 0
					rStericSum = rSteric[i]+rSteric[j];
					rSteric2 = np.power(rStericSum,2)
					if r2<rSteric2:
						r = np.sqrt(r2)
						irSteric = rStericSum/r
						fSteric = -wS[j]*(1-irSteric)
#                       pdb.set_trace()
						sP[eco.xC,i] += dx*fSteric 
						sP[eco.yC,i] += dy*fSteric

			
		fSlope = wS[i] #strength of the "slope" is given by the wS paramter of the particle

		sP[eco.yC,i] -= fSlope #currently spin is not normalized# Force in negative y direction
		
 
            #central force is located at the origin (otherwise add sx[i]-sxCentral etc.)
		#aligning force (Fxn) is the cross product of the total sum of forces on the particle and orientation 
		fAllign = wA[i]*(snx[i]*sP[eco.yC,i]-sny[i]*sP[eco.xC,i]) #n cross force
		sP[eco.nxC,i] +=  - sny[i]*fAllign
		sP[eco.nyC,i] +=  + snx[i]*fAllign#(sny[i] - snx[i]*fAllign)*normFAllign - sny[i]
			

		sP[eco.xC,i] += v0[i]*snx[i] #currently spin is not normalized
		sP[eco.yC,i] += v0[i]*sny[i]			
	
	

################################################# Phototaxiing Heterogeneous Interacting Population in a Periodic Box ##############################################
@numba.njit(parallel=True)
def _grPhototaxisInteractingPeriodic(eco,sp,sP,boxSize):
#Aligning particles 
### This is the main function to define the dynamics to propogate a heterogenenous population with alignment interaction, steric repulsion inherent speed.
### Here x,y, nx, ny are affected by forces etc.
### Size, v0, alignment and such are affected by "communication"
### The time change for each should be slightly different 
### DIfferent elements should be referenced through their respective position in sp table (say eco.sp[eco.xC,:]+=...)

	''' green function for only steric interactions with a heterogenous population and a central light spot for phototaxiing'''
	rTol = 1E-8; # (original)
	#rTol = 0.000633; ## Consistent with periodic boundary condition
	r2Tol = np.power(rTol,2);

	NN = eco.N;

	#periodicReg = 0.0015; #regularization parameter for smooth periodic boundary condition
	#periodicReg2 = np.power(periodicReg,2)
	# Assign shorter names for positions and orienations arrays:
	
	sx = sp[eco.xC,:]; # X coordinates
	sy = sp[eco.yC,:]; # Y coordinates
	
	snx = sp[eco.nxC,:]; # orientation x component
	sny = sp[eco.nyC,:]; # orientation y component

	
	wS = sp[eco.wSC,:];
	v0 = sp[eco.v0C,:];
	wA = sp[eco.wAC,:];
	rSteric = sp[eco.rStericC,:];

	#manually define the mobility - how much a particle is moved given the force it senses
	mobility = 1
	## The following fetch the parameters of the first particle thus treating the population as homogeneous.
	rStericCentral = boxSize/150*36*2/3.*1.2#rSteric[0] * 50; # Make a repulsive potential 20 times larger than particles

	## Calculate the squares for later reference
	rStericCentral2 = np.power(rStericCentral,2)
	
	for i in numba.prange(NN):
		for j in range(NN):
			if i != j:
				dx = sx[i] - sx[j]
				dy = sy[i] - sy[j]
				
				#find vectorial minimal separation on a torus
				dxA = np.abs(dx)
				dyA = np.abs(dy)
			
				
				if dxA > boxSize/2:
					dx = np.sign(dx)*(dxA-boxSize)
						
				if dyA > boxSize/2:
					dy = np.sign(dy)*(dyA-boxSize)
					
					
				r2 = np.power(dx,2) + np.power(dy,2)
				
				if r2 > r2Tol:

					fSteric = 0
					rStericSum = rSteric[i]+rSteric[j];
					rSteric2 = np.power(rStericSum,2)
					if r2<rSteric2:
						r = np.sqrt(r2)
						irSteric = rStericSum/r
						fSteric = -wS[j]*(1-irSteric)
#                       pdb.set_trace()
						sP[eco.xC,i] += dx*fSteric*mobility 
						sP[eco.yC,i] += dy*fSteric*mobility
		
		v = v0[i]
		
		#Calc distance from center of "light spot"
		rCentral2= np.power(sx[i]-boxSize/2,2)+np.power(sy[i]-boxSize/2,2) 
		
		if rCentral2 < rStericCentral2:
			v = 0
 
        #central force is located at the origin (otherwise add sx[i]-sxCentral etc.)
		#aligning force (Fxn) is the cross product of the total sum of forces on the particle and orientation 
		fAllign = wA[i]*(snx[i]*sP[eco.yC,i]-sny[i]*sP[eco.xC,i]) #n cross force
		sP[eco.nxC,i] +=  - sny[i]*fAllign
		sP[eco.nyC,i] +=  + snx[i]*fAllign#(sny[i] - snx[i]*fAllign)*normFAllign - sny[i]
			

		sP[eco.xC,i] += v*snx[i] #currently spin is not normalized
		sP[eco.yC,i] += v*sny[i]		
	

################################################# Phototaxiing Heterogeneous Interacting Population in a Periodic Box ##############################################
@numba.njit(parallel=True)
def _grPhototaxisSlowInLightInteractingPeriodic(eco,sp,sP,boxSize):
#Aligning particles 
### This is the main function to define the dynamics to propogate a heterogenenous population with alignment interaction, steric repulsion inherent speed.
### Here x,y, nx, ny are affected by forces etc.
### Size, v0, alignment and such are affected by "communication"
### The time change for each should be slightly different 
### DIfferent elements should be referenced through their respective position in sp table (say eco.sp[eco.xC,:]+=...)

	''' green function for only steric interactions with a heterogenous population and a central light spot for phototaxiing'''
	rTol = 1E-8; # (original)
	#rTol = 0.000633; ## Consistent with periodic boundary condition
	r2Tol = np.power(rTol,2);

	NN = eco.N;

	#periodicReg = 0.0015; #regularization parameter for smooth periodic boundary condition
	#periodicReg2 = np.power(periodicReg,2)
	# Assign shorter names for positions and orienations arrays:
	
	sx = sp[eco.xC,:]; # X coordinates
	sy = sp[eco.yC,:]; # Y coordinates
	
	snx = sp[eco.nxC,:]; # orientation x component
	sny = sp[eco.nyC,:]; # orientation y component

	
	wS = sp[eco.wSC,:];
	v0 = sp[eco.v0C,:];
	wA = sp[eco.wAC,:];
	rSteric = sp[eco.rStericC,:];

	#manually define the mobility - how much a particle is moved given the force it senses
	mobility = 1
	## The following fetch the parameters of the first particle thus treating the population as homogeneous.
	rStericCentral = boxSize/150*36*2/3.*1.2#rSteric[0] * 50; # Make a repulsive potential 20 times larger than particles

	## Calculate the squares for later reference
	rStericCentral2 = np.power(rStericCentral,2)
	
	for i in numba.prange(NN):
		for j in range(NN):
			if i != j:
				dx = sx[i] - sx[j]
				dy = sy[i] - sy[j]
				
				#find vectorial minimal separation on a torus
				dxA = np.abs(dx)
				dyA = np.abs(dy)
			
				
				if dxA > boxSize/2:
					dx = np.sign(dx)*(dxA-boxSize)
						
				if dyA > boxSize/2:
					dy = np.sign(dy)*(dyA-boxSize)
					
					
				r2 = np.power(dx,2) + np.power(dy,2)
				
				if r2 > r2Tol:

					fSteric = 0
					rStericSum = rSteric[i]+rSteric[j];
					rSteric2 = np.power(rStericSum,2)
					if r2<rSteric2:
						r = np.sqrt(r2)
						irSteric = rStericSum/r
						fSteric = -wS[j]*(1-irSteric)
#                       pdb.set_trace()
						sP[eco.xC,i] += dx*fSteric*mobility 
						sP[eco.yC,i] += dy*fSteric*mobility
		
		v = v0[i]
		
		#Calc distance from center of "light spot"
		rCentral2= np.power(sx[i]-boxSize/2,2)+np.power(sy[i]-boxSize/2,2) 
		
		if rCentral2 < rStericCentral2:
			v = v/15.
 
        #central force is located at the origin (otherwise add sx[i]-sxCentral etc.)
		#aligning force (Fxn) is the cross product of the total sum of forces on the particle and orientation 
		fAllign = wA[i]*(snx[i]*sP[eco.yC,i]-sny[i]*sP[eco.xC,i]) #n cross force
		sP[eco.nxC,i] +=  - sny[i]*fAllign
		sP[eco.nyC,i] +=  + snx[i]*fAllign#(sny[i] - snx[i]*fAllign)*normFAllign - sny[i]
			

		sP[eco.xC,i] += v*snx[i] #currently spin is not normalized
		sP[eco.yC,i] += v*sny[i]			
	
################################################################################################################################################
################################################# Container Class ##############################################################################
################################################################################################################################################

class greens():
#This class contains a few combinations of the Green's function with/without some time of repulsion. ,,
#grNumba - no repulsion
#grInteracintgHeteroPop - an interacting heteogeneous population
#grNonInteracintgHeteroPop - a non interacting heteorgeneous population
#grInteractingHeteroPopCentralPot - an interacting heterogeneous population with a central repulsive potential (obstacle)

#grPlusUSoftRepulsion - some sort of generalizes VdW
#grPlusExpRepulsion - exponential repulsion
#grPlusRepulsion - the repulsive part of a harmonic potential #this was the most stable strictly steric repulsion


#######################
	def grHeterInteractingoPop(eco,sp):#This may look redundent as eco already has sp, but numba does not support assignment inside class so have to be passed separately

		NN = eco.N;
		
		sP = np.zeros(np.shape(sp));
		_grHeteroPop(eco,sp,sP);

		return sP
	
	def grNonInteractingHeteroPopCentralPot(eco,sp):
		'''
		Propagates an ecosystem of a heterogenous system.
		Particles are in an open doamin.
		Particles are non-interacting.
		Particles align with force according to Ben Zion et al arXiv 2022
		'''
		NN = eco.N;
		
		sP = np.zeros(np.shape(sp));
		_grNonInteractingHeteroPop(eco,sp,sP);

		return sP
	
	
	def grInteractingHeteroPopPeriodic(eco,sp,boxSize):
		'''
		Propagates an ecosystem of a heterogenous system.
		Particles are in a periodic box of size boxSize.
		All all particles interact with all other particles through a soft core repulsion.
		Particles align with force according to Ben Zion et al arXiv 2022
		'''
		NN = eco.N;
		
		sP = np.zeros(np.shape(sp)); # note that this is a funny place to create a large matrix of zeros but experimenting with having a single matrix througout that is re-zeroed showed similar prefromence (so probably numpy is very efficient about it)
		_grInteractingHeteroPopPeriodic(eco,sp,sP,boxSize);

		return sP		
	def grInteractingHeteroPopPeriodicOptimized(eco, sp, boxSize):
		NN = eco.N;
		
		sP = np.zeros(np.shape(sp)); # note that this is a funny place to create a large matrix of zeros but experimenting with having a single matrix througout that is re-zeroed showed similar prefromence (so probably numpy is very efficient about it)
		_grInteractionHeteroPopPeriodicOptimized(eco,sp,sP,boxSize);
	
		return sP
	
	def grNonInteractingHeteroWithPassivePopPeriodic(eco,sp,boxSize):
		'''
		Propagates an ecosystem of a heterogenous system.
		Particles are in a periodic box of size boxSize.
		Particles have a soft-core repusion.
		Running the following interaction scheme:
			Active-active: non interacting
			Active-passive: interacting
			Passive-passive: interacting
		Particles align with force according to Ben Zion et al arXiv 2022
		'''
		NN = eco.N;
		
		sP = np.zeros(np.shape(sp)); # note that this is a funny place to create a large matrix of zeros but experimenting with having a single matrix througout that is re-zeroed showed similar prefromence (so probably numpy is very efficient about it)
		_grNonInteractingHeteroWithPassivePopPeriodic(eco,sp,sP,boxSize);

		return sP		
	
	def grInteractingHeteroPopCentralPot(eco,sp):
		'''
		Propagates an ecosystem of a heterogenous, non-interacting active particles.
		Particles have a soft-core repusion.
		Particles are subjected to a repulsive, soft-core, central potenial.
		Particles are in an unbounded domain.
		Particles align with force according to Ben Zion et al arXiv 2022
		'''

		NN = eco.N;
		
		sP = np.zeros(np.shape(sp));
		_grInteractingHeteroPopCentralPot(eco,sp,sP);
		

		return sP
	

	def grInteractingHeteroPopConstantForce(eco,sp):
	
		'''
		Propagates an ecosystem of a heterogenous, interacting active particles.
		Particles have a soft-core repusion.
		Particles are subjected to a constant force along the y direction.
		Particles are in an unbounded domain.
		Particles align with force according to Ben Zion et al arXiv 2022
		'''

		NN = eco.N;
		
		sP = np.zeros(np.shape(sp));
		_grInteractingHeteroPopCentralPot(eco,sp,sP);
		

		return sP

	
	def grPhototaxisInteractingPeriodic(eco,sp,boxSize):
		'''
		Propagates an ecosystem of a heterogenous system.
		Particles are in a periodic box of size boxSize.
		All all particles interact with all other particles through a soft core repulsion.
		Particles align with force according to Ben Zion et al arXiv 2022
		Particles' self propulsion speed is set to 0 at  central circle (light spot)
		'''
		NN = eco.N;
		
		sP = np.zeros(np.shape(sp)); # note that this is a funny place to create a large matrix of zeros but experimenting with having a single matrix througout that is re-zeroed showed similar prefromence (so probably numpy is very efficient about it)
		_grPhototaxisInteractingPeriodic(eco,sp,sP,boxSize);

		return sP		

	def grPhototaxisSlowInLightInteractingPeriodic(eco,sp,boxSize):
		'''
		Propagates an ecosystem of a heterogenous system.
		Particles are in a periodic box of size boxSize.
		All all particles interact with all other particles through a soft core repulsion.
		Particles align with force according to Ben Zion et al arXiv 2022
		Particles' self propulsion speed is set to 1/15 of nominal at  central circle (light spot)
		'''
		NN = eco.N;
		
		sP = np.zeros(np.shape(sp)); # note that this is a funny place to create a large matrix of zeros but experimenting with having a single matrix througout that is re-zeroed showed similar prefromence (so probably numpy is very efficient about it)
		_grPhototaxisSlowInLightInteractingPeriodic(eco,sp,sP,boxSize);

		return sP	
	
	def grInteractingHeteroPopCircularWall(eco, sp, boxSize):
		"""
		Propagates an ecosystem of a heterogeneous population.
		Particles are confined inside a circular wall of radius boxSize / 2.
		All particles interact with all others through soft-core steric repulsion.
		Self-propulsion is applied along the current orientation.
		"""
		NN = eco.N;

		sP = np.zeros(np.shape(sp));
		_grInteractingHeteroPopCircularWall(eco, sp, sP, boxSize);
		return sP;

	def grInteractingHeteroPopRectangularWall(eco, sp, boxSize):
		"""
		Propagates an ecosystem of a heterogeneous population.
		Particles are confined in a rectangular domain with soft repulsive boundaries.
		Interactions include pairwise steric repulsion and alignment torque.
		Particles self-propel in the direction of their orientation.
		"""
		NN = eco.N
		sP = np.zeros(np.shape(sp))
		_grInteractingHeteroPopRectangularWall(eco, sp, sP, boxSize)
		return sP;
