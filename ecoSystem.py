####################################################
########### Ecosystem class ##############
####################################################

### Written by Matan Yah Ben Zion and Naomi Oppenheimer 2020 05 10 
### Updated by Matan Yah Ben Zion 2022 05 24 

#### This class has an ecosystem of heterogeneous active particles.
#### The ecosystem consists of N particles with the following characterisits:
#### Location (xC,yC), orientation (nxC,nyC), speed (v0C), steric radius (rStericC), steric repulsion strength (wSC) 
#### The data is structured as a matrix in 'self.sp' where the first element is the property and the second element is the particle number.
#### e.g the y orientation of particle number 100 is self.sp[5,100].
#### The class is designed to be implemented using NUMBA


### To change the number of paratmer:
### Add each new param in spec
### Change the number for initial value of params in __init__ function
### Add approriate index for each param inside the __init__ function


'''
TODO: add everything from randomActiveWithBoundary file to make a simulation engine.
'''
#from numba import jitclass          # import the decorator
from numba.experimental import jitclass
from numba import uint, uint16, int16, int32, float32    # import the types
import numpy as np

def l2n(x):
	return np.array(x);

# Specs is defined to be passed through jit for numba compatibitliy.
spec = [
	('N', int32), # Number of particles in the population
	('xC', uint16), # Particles x coordinates
	('yC', uint16), # Particles y coordinates
	('nxC', uint16), # Partircles orientation along x
	('nyC', uint16), # Partircles orientation along y
	('wSC', uint16),
	('rStericC', uint16),
	('v0C', uint16),
	('wAC', uint16),
	('sp', float32[:,:])
]

@jitclass(spec)
class ecoSystem():
	def __init__(self, N,params=8):
		# number of parameters describing a particle (x,y,nx,ny,rSteric,wS,v0 etc.) x=0, y=0, rSteric=0.01, wS=5E4, nx=1, ny=0,v0=1):
		self.N = N;
		# The indexes of the different parameters to be referred in sp matrix
		self.xC = 0;
		self.yC = 1;
		self.nxC = 2;
		self.nyC = 3;
		self.wSC = 4;		
		self.rStericC = 5;
		self.v0C = 6;
		self.wAC = 7;

		self.sp = np.zeros((params,1*self.N),dtype=np.float32);#range(10)#np.zeros((7,N));

	def initializeEcosystem(self,boxSize, rSteric=0.01,wS=5E4,v0=1,wA=0):
        #Define the radius, interactions strength, speed of particles.
        #Then sets their intial positions and orienations.
		self.initializeParticlesParams(rSteric=rSteric, wS=wS, v0=v0,wA=wA);
		self.initializeRandomPositions(boxSize);
		self.initializeRandomOrientations();

	def initializeParticlesParams(self, rSteric=0.01,wS=5E4,v0=1,wA=0): 
		#Generally speaking, can pass here an array length number of particles or a scalar
		self.sp[self.rStericC,:] = np.ones(1*self.N)*rSteric;
		self.sp[self.wSC,:] = np.ones(1*self.N)*wS;
		self.sp[self.wAC,:] = np.ones(1*self.N)*wA;
		self.sp[self.v0C,:] = np.ones(1*self.N)*v0;

	def initializeRandomPositions(self, boxSize):
		cx, cy = boxSize / 2, boxSize / 2
		rStericWall = self.sp[self.rStericC, 0] * 50
		confined_radius = rStericWall * 0.95  # stay inside the wall
		angles = np.random.uniform(0, 2 * np.pi, self.N)
		radii = np.sqrt(np.random.uniform(0, 1, self.N)) * confined_radius
		self.sp[self.xC, :] = cx + radii * np.cos(angles)
		self.sp[self.yC, :] = cy + radii * np.sin(angles)

	def initializeRandomOrientations(self):
		#initial particles' orientations randomly
		self.sp[self.nxC,:] = np.random.randn(1*self.N);
		self.sp[self.nyC,:] = np.random.randn(1*self.N);
		#normalize the orientation vectors:
		self.normalizeOrientations()
		
	def updateEcoSystem(self,newEco):
		self.sp=newEco
		
	def normalizeOrientations(self):
		#Normalize the orientation vector (this is a fast normalization yet implementation is combersome as sp indices should not be written explicitly)
		invN = 1/np.sqrt(np.power(self.sp[self.nxC,:],2)+np.power(self.sp[self.nyC,:],2))
		self.sp[self.nxC,:] *=invN;
		self.sp[self.nyC,:] *=invN;
		
		
		