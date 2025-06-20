####### Time propagation of simulation using 5th order Runge Kutta #######
################# Written by Matan and Naomi 2020 05 20 ##################

#### This class is the core of the simulation engine
#### It receives an ecosystem in a form of a matrix where each row is a particle with its characteristics (position, orienation, size, speed etc.)
#### It then propagates in time the next position of each particle in the ecosystem

### Updated 20220527 by Matan to allow periodic boundary condition

import numpy as np
import pickle
from scipy.stats import norm

class timePropagation():
	'''
	Time propagation wrapper. Propogates an eco system of particles.
	Particles parameters (position, inherent velocity etc) are defined in ps matrix (paramsXN)
	Indices for for referencing the parameters are in eco class (eco.xC, eco.vC etc...)
	once propogates, runs a 5th order runge kuta algorithm using the selected green fuction.
	During the simulation tSave equilispaced snapshots are saved to tArray files on the hard-drive  (usualy 10 files)
	'''
    
	def __init__(self, dt,eco,fileBaseName):
		self.dt = dt
		self.eco = eco
		self.fileBaseName = fileBaseName
		
		self.Z = np.zeros(np.shape(self.eco.sp))

    
	def timeProp5Runge(self, T, tSave, tArray, greenFunc, kT, particlesToSave=None):
		'''
		T: number of time steps
		tSave: number of snap shots to save
		tArray: number of files to which save the snapshots equally distributed along simulation
		greenFunc: interaction rules propagator
		kT: temperature for random noise
		particlesToSave: list of particle indices to save (e.g., [-1] for last particle, or range(N) for all)
		'''
		self.T = T
		self.tSave = tSave
		self.tArray = tArray
		self.kT = kT

		eco = self.eco
		sp = eco.sp.copy()
		N = sp.shape[1]

		# Handle particle selection
		if particlesToSave is None:
			particlesToSave = list(range(N))
		elif particlesToSave == [-1]:
			particlesToSave = [N - 1]

		self.particlesToSave = particlesToSave
		nSave = len(particlesToSave)

		self.SP = np.zeros((int(tSave / tArray), sp.shape[0], nSave))
		self.addNoise = np.zeros(np.shape(sp))

		# Runge-Kutta 5(4) coefficients
		b1 = 35. / 384; b2 = 0.; b3 = 500. / 1113; b4 = 125. / 192
		b5 = -2187. / 6784; b6 = 11. / 84
		a21 = 1. / 5
		a31 = 3. / 40; a32 = 9. / 40
		a41 = 44. / 45; a42 = -56. / 15; a43 = 32. / 9
		a51 = 19372. / 6561; a52 = -25360. / 2187; a53 = 64448. / 6561; a54 = -212. / 729
		a61 = 9017. / 3168; a62 = -355. / 33; a63 = 46732. / 5247; a64 = 49. / 176; a65 = -5103. / 18656

		for ttt in range(self.tArray):
			for tt in range(int(tSave / tArray)):
				self.SP[tt] = sp[:, particlesToSave]

				for t in range(int(T / tSave)):
					SP0 = sp.copy()

					self.Z = greenFunc(eco, sp)
					k1 = self.dt * self.Z
					sp = SP0 + a21 * k1

					self.Z = greenFunc(eco, sp)
					k2 = self.dt * self.Z
					sp = SP0 + a31 * k1 + a32 * k2

					self.Z = greenFunc(eco, sp)
					k3 = self.dt * self.Z
					sp = SP0 + a41 * k1 + a42 * k2 + a43 * k3

					self.Z = greenFunc(eco, sp)
					k4 = self.dt * self.Z
					sp = SP0 + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4

					self.Z = greenFunc(eco, sp)
					k5 = self.dt * self.Z
					sp = SP0 + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5

					self.Z = greenFunc(eco, sp)
					k6 = self.dt * self.Z

					advance = (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)

					self.scaleNoise = np.sqrt(2.0 * self.dt * self.kT)

					noiseTable = norm.rvs(size=np.shape(sp[:4, :]), scale=self.scaleNoise)
					self.addNoise[:2, :] = noiseTable[:2]
					self.addNoise[2:4, :] = noiseTable[2:4] * 5

					sp = SP0 + advance + self.addNoise

					# Normalize orientation vectors
					invN = 1 / np.sqrt(sp[eco.nxC, :] ** 2 + sp[eco.nyC, :] ** 2)
					sp[eco.nxC, :] *= invN
					sp[eco.nyC, :] *= invN

			# Save file
			fileName = self.fileBaseName + '_' + str(ttt).zfill(3) + '_of_' + str(self.tArray).zfill(3) + '.pkl'
			with open(fileName, 'wb') as fHand:
				pickle.dump(self.SP, fHand)

			
			
	def timeProp5RungePeriodic(self,T,tSave,tArray,greenFunc,kT,boxSize,particlesToSave):
		'''
		T: number of time steps
		tSave: number of snap shots to save
		tArray: number of files to which save the snapshots
		greenFunc: interaction rules propogator
		kT: temperature for random noise
		'''
		self.T = T
		self.tSave = tSave # How many snap shots to take from the simulation
		self.tArray = tArray # How many files to save those snapshots equily distributed along simulation
		self.kT = kT #Temperature in arbitrary units
		#self.R = R

		eco = self.eco;		
		sp = eco.sp.copy();
		
		self.SP = np.zeros( np.concatenate( ([np.int_(self.tSave/self.tArray)],np.shape(sp))) )

		self.addNoise = np.zeros(np.shape(sp));
		"""
		Using fifth order Runge-Kutta with constant step size 				
		""" 
		b1 = 35./384; b1s = 5179./57600; b2 = 0.; b2s = 0.; b3 = 500./1113; b3s = 7571./16695; 
		b4 = 125./192; b4s = 393./640; b5 = -2187./6784; b5s = -92097./339200; b6 = 11./84; b6s = 187./2100; 
		b7s = 1./40; 
		a21 = 1./5; 
		a31 = 3./40; a32 = 9./40; 
		a41 = 44./45; a42 = -56./15; a43 = 32./9;
		a51 = 19372./6561; a52 = -25360./2187; a53 = 64448./6561; a54 = -212./729;
		a61 = 9017./3168; a62 = -355./33; a63 = 46732./5247; a64 = 49./176; a65 = -5103./18656;

		for ttt in range(self.tArray):
			for tt in range(np.int_(self.tSave/self.tArray)):
				self.SP[tt] = sp

				for t in range(np.int_(self.T/self.tSave)):
					SP0 = sp.copy()

					self.Z = greenFunc(eco,sp,boxSize) #Numba requires eco and sp to be passed separately because later assignment can not be done to class
					k1 = self.dt*self.Z 
					sp = SP0 + a21*k1;   #this assignment and those that follow can not be done inside the eco class while using NUMBA

					self.Z = greenFunc(eco,sp,boxSize)
					k2 = self.dt*self.Z
					sp = SP0 + a31*k1 + a32*k2; 

					self.Z = greenFunc(eco,sp,boxSize)
					k3 = self.dt*self.Z
					sp = SP0 + a41*k1 + a42*k2 + a43*k3;

					self.Z = greenFunc(eco,sp,boxSize)
					k4 = self.dt*self.Z
					sp = SP0 + a51*k1 + a52*k2 + a53*k3 + a54*k4;

					self.Z = greenFunc(eco,sp,boxSize)
					k5 = self.dt*self.Z
					sp = SP0 + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5;

					self.Z = greenFunc(eco,sp,boxSize)
					k6 = self.dt*self.Z

					advance = (b1*k1+b2*k2+b3*k3+b4*k4+b5*k5+b6*k6) #the value of a step according to fifth order Runge Kuta

					self.scaleNoise = np.sqrt(2.0*self.dt*self.kT)  # COMMENT - this is where fast and slow vars differ - the noise is for x, y, nx, ny

					
					
					#noise for translation
					positionNoise = False
					if positionNoise :
						noiseTablePosition = norm.rvs(size=np.shape(sp[:2,:]), scale=self.scaleNoise);
						self.addNoise[:2,:] =  noiseTablePosition; # noise for position
					
					#noise for orientation
					orientationNoise = True
					if orientationNoise:
						noiseTableOrientation = norm.rvs(size=np.shape(sp[2:4,:]), scale=self.scaleNoise);
						self.addNoise[2:4,:] = noiseTableOrientation; # noise for orientation


					sp= SP0 + advance + self.addNoise

					
				
					#Normalize the orientation vector:

					invN = 1/np.sqrt(np.power(sp[eco.nxC,:],2)+np.power(sp[eco.nyC,:],2))
					sp[eco.nxC,:] *=invN;
					sp[eco.nyC,:] *=invN;
					
			
					#no slip boundary conditions?
					#self.rr = np.sqrt(self.x[0]**2+self.x[1]**2)
					#sR = (self.rr>=self.R)
					#nsR = np.logical_not(sR)
					#self.x[:,nsR] = self.x[:,nsR] - advance[:,nsR]
#save the file
			fileName = self.fileBaseName+'_'+str(ttt).zfill(3)+'_of_'+str(self.tArray).zfill(3)+'.pkl'
			fHand = open(fileName, 'wb')
			pickle.dump(self.SP[:,:,particlesToSave],fHand)
			fHand.close()