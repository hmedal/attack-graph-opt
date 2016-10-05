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
import attackGraph
from mpi4py import MPI

def readMasterModel(filename):
    myMasterModel = read()
    theta = myMasterModel.getVars()[0]
    x = myMasterModel.getVars()[1:numArcs] # change the indices
    beta = myMasterModel.getVars()[1:numArcs] # change the indices
    B = myMasterModel.getVars()[1:numArcs] # change the indices

def readMasterModel(filename):
    pass # @TODO-Tanveer: implement this method

def solveSubproblems():
    print "solving subproblems on rank " + str(rank)
    pass # @TODO-Tanveer: implement this method

def modifyMasterModel():
    pass # @TODO-Tanveer: implement this method

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a filename.')
    parser.add_argument('-m','--masterproblem', help='the master problem file')
    parser.add_argument('-s','--mastersolution', help='the file containing the solution to the master problem')
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    numprocs = comm.size
    rank =comm.Get_rank()
    readMasterModel(args.masterproblem)
    readMasterSolution(args.mastersolution)
    solveSubproblems()
    modifyMasterModel()
    MPI.Finalize()    
    
