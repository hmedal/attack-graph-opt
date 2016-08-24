from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    data = [1,1,1]
else:
    data = None

data = comm.bcast(data, root=0)
print('rank',rank,data)