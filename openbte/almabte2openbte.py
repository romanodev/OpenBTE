import numpy as np
import sys
from openbte.utils import save
import argparse


def almabte2openbte():
 """Read AlmaBTE Data"""
 
 #Parse filename
 parser = argparse.ArgumentParser()
 parser.add_argument('-f',  help='filename',default = '')
 parser.add_argument('-o',  help='output',default = 'rta')
 args = parser.parse_args()
 #-------------

 tmp = np.loadtxt(args.f,skiprows=1,delimiter=',')
 nq = int(np.max(tmp[:,0]))+1
 nb = int(np.max(tmp[:,1]))+1
 ntot = nq*nb

 tau   = tmp[:,7 ]
 v     = tmp[:,8:]
 C     = tmp[:,6 ]
 w     = tmp[:,5 ]
 kappa =  np.einsum('ki,kj,k,k->ij',v,v,tau,C)

 data = {'heat_capacity':C,\
         'scattering_time':tau,\
         'group_velocity':v,\
         'frequency':w/2.0/np.pi,\
         'thermal_conductivity':kappa}

 save(args.o,data)


