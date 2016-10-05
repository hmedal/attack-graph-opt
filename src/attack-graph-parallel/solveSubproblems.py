'''
Created on Sep 30, 2016

@author: hm568
'''
import random
import datetime
from gurobipy import *
import copy
import math
import numpy as np
from mpi4py import MPI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a filename.')
    parser.add_argument('-m','--masterproblem', help='the master problem file')
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    numprocs = comm.size
    rank =comm.Get_rank()
    
