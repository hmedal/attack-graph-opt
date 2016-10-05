import os
import argparse
from gurobipy import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
numprocs = comm.size
rank =comm.Get_rank()
    
if rank == 0:
    os.system('mpirun -np 10 python helloworld.py') # solve master problem
    
else:
    print 'my rank is ' + str(rank)