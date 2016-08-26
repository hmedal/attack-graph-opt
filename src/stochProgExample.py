import numpy
from mpi4py import MPI

class MasterProblem(object):
    
    # probably want to create the model in the __init__ method
    
    def solve(self):
        pass
    
    def getOptimalSolution(self):
        # needs more here
        return [1, 1, 1] # change this later
    
class SecondStageProblem(object):
    
    # probably want to create the model in the __init__ method
    
    def setFirstStageSolution(self, firstStageSolution):
        self.firstStageSolution = firstStageSolution
    
    def setRighthandSide(self, rhs):
        self.rhs = rhs
    
    def getOptimalObjectiveValue(self):
        # needs more here
        return self.rhs + sum(self.firstStageSolution) # change this later
    
secondStageProb = SecondStageProblem() # create second stage problem here

def computeSecondStageObjValue(firstStageSolution,rhs):
    secondStageProb.setFirstStageSolution(firstStageSolution)
    secondStageProb.setRighthandSide(rhs)
    return secondStageProb.getOptimalObjectiveValue()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# initialization
probabilities = [1.0/size for x in range(size)] # change this later
firstStageSolution = []
if rank == 0:
    masterProblem = MasterProblem()
    rhsValues = [x for x in range(size)]
else:
    rhsValues = None
    firstStageSolution = None

# starting the algorithm
if rank == 0:
    masterProblem.solve()
    firstStageSolution = masterProblem.getOptimalSolution()
else:
    firstStageSolution = None

i = 0
while(True):
    firstStageSolution = comm.bcast(firstStageSolution, root = 0)
    rhsValue = comm.scatter(rhsValues, root=0)
    secondStageObjectiveValue = computeSecondStageObjValue(firstStageSolution,rhsValue)
    print('iteration', i, 'rank',rank,'has second Stage Objective Value:', secondStageObjectiveValue, rhsValue)
    
    secondStageObjectiveValues = comm.gather(secondStageObjectiveValue,root=0)
    
    if rank == 0:
        expectedCost = sum([a * b for a,b in zip(secondStageObjectiveValues, probabilities)])
        print('iteration', i, 'expected Cost:', expectedCost)
    if i >= 3:
        break
    i += 1