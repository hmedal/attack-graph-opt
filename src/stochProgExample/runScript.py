import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a filename.')
    parser.add_argument('-m','--master', help='the master node')
    parser.add_argument('-w', '--worker_pool', help='the worker pool servers')
    parser.add_argument('-n', '--count', help='number of worker pool servers')
    args = parser.parse_args()
    numProcs = args.count*20
    max_iter = 10
    for iter in range(1,max_iter + 1):
        os.system('mpirun -n 1 -hosts ' + args.master + ' gurobi_cl WorkerPool=' + args.worker_pool + ' DistributedMIPJobs=' + args.count + ' ResultFile=' + 'master_' + iter + '.sol' + ' /usr/local/gurobi652/linux64/examples/data/misc07.mps') # solve master problem
        os.system('mpirun -np' + str(numProcs) + ' python subProbs.py master_' + str(iter) + '.sol') # solve sub problem
        # read upper bound from a file
        # read lower bound from a file
        # check termination; print final solution and objective value to file