from mpi4py import MPI

def computeSecondStageObjValue(firstStageSolution,rhs):
   # insert code for solving second stage
   return rhs + sum(firstStageSolution)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

probabilities = [1.0/size for x in range(size)]
firstStageSolution = [1,1,1]
if rank == 0:
   rhsValues = [x for x in range(size)]
   print('we will be scattering:', rhsValues)
else:
   rhsValues = None
   
rhsValue = comm.scatter(rhsValues, root=0)
secondStageObjectiveValue = computeSecondStageObjValue(firstStageSolution,rhsValue)
print('rank',rank,'has secondStageObjectiveValue:', secondStageObjectiveValue)

newData = comm.gather(secondStageObjectiveValue,root=0)

if rank == 0:
   expectedCost = sum([a * b for a,b in zip(newData, probabilities)])
   print('expectedCost:', expectedCost)