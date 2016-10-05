import os
import argparse

max_iter = 10
tolerance= 0.05 #gap between the upper and lower bound

def getLB(filename):
    file = open(filename)
    ub = float(file.readlines()[1].split()[4])
    file.close
    return ub

def createAndWriteInitialMasterProblemModel():
    theta = masterModel.addVar(lb=0, ub=GRB.INFINITY, obj=1, name='theta')
    Bd= 150.0           #Defender's budget
    A = []      #List of all arcs
    ##Define the interdiction variable
    x = {}
    for a in range(len(A)):
        x[a] = masterModel.addVar(vtype=GRB.BINARY, name='x'+str(A[a].tail)+str(A[a].head))
        ##Update the model
    masterModel.update()
    masterModel.addConstr(quicksum(A[a].securityCost* x[a] for a in range(len(A))) <= Bd)
    masterModel.write("masterModel.lp")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a filename.')
    parser.add_argument('-m','--master', help='the master node')
    parser.add_argument('-w', '--worker_pool', help='the worker pool servers')
    parser.add_argument('-n', '--count', help='number of worker pool servers')
    parser.add_argument('-d', '--datafile', help='the data file')
    args = parser.parse_args()
    numProcs = args.count*20
    createAndWriteInitialMasterProblemModel()
    ub = float('inf')
    lb = 0
    while UB-LB >= tolerance * UB:
        masterProblemFile = 'masterModel.lp'
        os.system('mpirun -n 1 -hosts ' + args.master + ' gurobi_cl WorkerPool=' + args.worker_pool + ' DistributedMIPJobs=' + args.count + ' ResultFile=' + 'master_' + str(iter) + '.sol' + ' ' + masterProblemFile) # solve master problem
        os.system('mpirun -np ' + str(numProcs) + ' python solveSubproblems.py master_' + str(iter) + '.sol') # solve sub problem, given solution to master problem; change "subProbs.py"
        LB = getLB("master_" + str(iter) + ".sol")
        # @TODO-Tanveer: read upper bound from a file or compute it somehow
        if hasTerminated(iter, ub, lb):
    #@TODO-Tanveer: print final solution and objective value to file