import numpy
from mpi4py import MPI

class MasterProblem(object):
    
    def solve(self):
        pass
    
    def getOptimalSolution(self):
        return [1, 1, 1] # change this later
    
class SecondStageProblem(object):
    
    def setFirstStageSolution(self, firstStageSolution):
        self.firstStageSolution = firstStageSolution
    
    def setRighthandSide(self, rhs):
        self.rhs = rhs
    
    def getOptimalObjectiveValue(self):
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
probabilities = [1.0/size for x in range(size)]
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

firstStageSolution = comm.bcast(firstStageSolution, root = 0)
rhsValue = comm.scatter(rhsValues, root=0)
secondStageObjectiveValue = computeSecondStageObjValue(firstStageSolution,rhsValue)
print('rank',rank,'has second Stage Objective Value:', secondStageObjectiveValue, rhsValue)

secondStageObjectiveValues = comm.gather(secondStageObjectiveValue,root=0)

if rank == 0:
    expectedCost = sum([a * b for a,b in zip(secondStageObjectiveValues, probabilities)])
    print('expected Cost:', expectedCost)