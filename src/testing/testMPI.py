import os
import argparse
from gurobipy import *
from mpi4py import MPI

tolerance= 0.05 #gap between the upper and lower bound
Bd= 150.0           #Defender's budget

comm = MPI.COMM_WORLD
numprocs = comm.size
rank =comm.Get_rank()

def createAndWriteInitialMasterProblemModel():
    masterModel = Model("master")
    theta = masterModel.addVar(lb=0, ub=GRB.INFINITY, obj=1, name='theta')
    A = []      #List of all arcs
    ##Define the interdiction variable
    x = {}
    for a in range(len(A)):
        x[a] = masterModel.addVar(vtype=GRB.BINARY, name='x'+str(A[a].tail)+str(A[a].head))
        ##Update the model
    masterModel.update()
    masterModel.addConstr(quicksum(A[a].securityCost* x[a] for a in range(len(A))) <= Bd)
    masterModel.write("masterModel.lp")

parser = argparse.ArgumentParser(description='Read a filename.')
parser.add_argument('-m','--master', help='the master node')
parser.add_argument('-w', '--worker_pool', help='the worker pool servers')
parser.add_argument('-n', '--count', help='number of worker pool servers')
parser.add_argument('-d', '--datafile', help='the data file')
args = parser.parse_args()
numProcs = args.count*20
createAndWriteInitialMasterProblemModel()
UB = float('inf')
LB = 0
    
if rank == 0:
    masterProblemFile = 'masterModel.lp'
    os.system('mpirun -n 1 -hosts ' + args.master + ' gurobi_cl WorkerPool=' + args.worker_pool + ' DistributedMIPJobs=' + args.count + ' ResultFile=' + 'master_' + str(iter) + '.sol' + ' ' + masterProblemFile) # solve master problem
    
else:
    print 'my rank is ' + str(rank)