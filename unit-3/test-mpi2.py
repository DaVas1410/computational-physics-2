from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    data = {'a': 1, 'b': 2, 'c': 3}
    value_to_broadcast = data['a']
else:
    value_to_broadcast = None

received_value = comm.bcast(value_to_broadcast, root=0)
print('rank', rank, received_value)
