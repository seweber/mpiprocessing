import time
from mpi4py import MPI
import pickle
import sys
import os

if __name__ == "__main__":
    # Set MPI environment variables
    status = MPI.Status()
    host = MPI.Get_processor_name()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nProcesses = comm.Get_size()
    
    # Tags
    WORKTAG = 0
    DIETAG = 1
    WORKERTAG = 2
    MASTERTAG = 3
    ERRORTAG = 4
    
    ######################################################################
    ### Determine who is master and who is slave #########################
    ######################################################################
    
    host_master, path_lock = sys.argv[1:3]
    if host == host_master:
        if os.name == 'nt':
            if rank == 0: # TODO use similar approach as for linux as rank 0 is not necessarily on host_master
                rank_master = rank
                rank_slaves = [r for r in range(0,nProcesses) if r != rank_master]
                for j in rank_slaves:
                    comm.isend(obj=rank_master, dest=j, tag=MASTERTAG)
            else:
                rank_master = comm.recv(source=MPI.ANY_SOURCE, tag=MASTERTAG,
                                        status=status)
            comm.barrier()
        else:
            import fcntl
            with open(path_lock, "w") as fd:
                try:
                    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    rank_master = rank
                    rank_slaves = [r for r in range(0,nProcesses) if r != rank_master]
                    for j in rank_slaves:
                        comm.isend(obj=rank_master, dest=j, tag=MASTERTAG)
                except BlockingIOError:
                    rank_master = comm.recv(source=MPI.ANY_SOURCE, tag=MASTERTAG,
                                            status=status)
                comm.barrier()
    else:
        rank_master = comm.recv(source=MPI.ANY_SOURCE, tag=MASTERTAG, status=status)
        comm.barrier()

    if rank_master == rank:
        
        ######################################################################
        ### Master ###########################################################
        ######################################################################
        
        import mmap
        import struct
        
        # Helper functions
        def receivOutput():
            result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            dest = status.Get_source()
            tag = status.Get_tag()
            return result, dest, tag
        
        def setflag(flag):
            mm_flags.seek(0)
            mm_flags.write(str(flag).encode())
        
        def waitflag(flag):
            while True:
                mm_flags.seek(0)
                string = mm_flags.read()
                if int(string.decode()) >= flag: break
        
        # Get paths
        path_flags, path_work, path_worker, path_result = sys.argv[3:]

        # Open memory maps
        f_flags = open(path_flags, "r+b")
        f_work = open(path_work, "rb")
        f_worker = open(path_worker, "rb")
        f_result = open(path_result, "r+b")
        
        if os.name == 'nt':
            mm_flags = mmap.mmap(f_flags.fileno(), 0, access=mmap.ACCESS_READ|mmap.ACCESS_WRITE)
            mm_work = mmap.mmap(f_work.fileno(), 0, access=mmap.ACCESS_READ)
            mm_worker = mmap.mmap(f_worker.fileno(), 0, access=mmap.ACCESS_READ)
            mm_result = mmap.mmap(f_result.fileno(), 0, access=mmap.ACCESS_WRITE)
        else:
            mm_flags = mmap.mmap(f_flags.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_READ|mmap.PROT_WRITE)
            mm_work = mmap.mmap(f_work.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_READ)
            mm_worker = mmap.mmap(f_worker.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_READ)
            mm_result = mmap.mmap(f_result.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_WRITE)

        # Get worker
        waitflag(1)
        mm_worker.seek(0)
        is_imap = struct.unpack("?",mm_worker.read(1))[0]
        sz = struct.unpack("L",mm_worker.read(8))[0]
        comm.bcast(obj=mm_worker.read(sz), root=rank_master)

        # Get work
        waitflag(2)
        mm_work.seek(0)
        params = pickle.load(mm_work)
                
        # Set variables and buffers
        params          = list(params)
        nWork           = len(params)            
        results         = [None]*nWork
        worker2idx      = [None]*nProcesses
     
        # Helper for saving results
        def save(result, results, worker2idx, dest, j1):
            # Save to cache
            results[worker2idx[dest]] = result
                    
            # Save part of results if is_imap
            if is_imap:
                j2 = j1
                for r in results[j1:]:
                    if r is None: break
                    j2 += 1
                if j2 > j1:
                    mm_result.seek(0)
                    pickle.dump(results[j1:j2], mm_result)
                    setflag(3)
                    results[j1:j2] = [None]*(j2-j1)
                    j1 = j2
                    waitflag(4)
            
            return results, j1
        
        # Loop over the work
        nWorking = 0
        j1 = 0
        for idx, work in enumerate(params):
            if idx < len(rank_slaves):
                # Save which worker will obtain the next input
                dest = rank_slaves[idx]
                nWorking += 1
                
                # Send the input to the worker & save which indices the worker received
                worker2idx[dest] = idx
                comm.isend(obj=work, dest=dest, tag=WORKTAG)
        
            elif nWorking > 0:
                # Receive the output from the worker & save which worker has send the output and will obtain the next input
                result, dest, tag = receivOutput()
                
                if tag != ERRORTAG:
                    # Save result
                    results, j1 = save(result, results, worker2idx, dest, j1)
                    
                    # Send the input to the worker & save which indices the worker received
                    worker2idx[dest] = idx
                    comm.isend(obj=work, dest=dest, tag=WORKTAG)
                else:
                    nWorking -= 1
                    
        # Collect - until now unreceived - results
        for _ in range(nWorking):
            result, dest, tag = receivOutput()
            if tag != ERRORTAG:
                # Save result
                results, j1 = save(result, results, worker2idx, dest, j1)
        
        # Save results if not is_imap
        if not is_imap:
            mm_result.seek(0)
            pickle.dump(results, mm_result)
            setflag(3)
        else:
            setflag(5)
        
        # Terminate all slaves
        for j in rank_slaves:
            comm.isend(obj=None, dest=j, tag=DIETAG)
            
        # Close memory maps
        mm_flags.close()
        f_flags.close()
        mm_work.close()
        f_work.close()
        mm_worker.close()
        f_worker.close()
        mm_result.close()
        f_result.close()
    
    else:
        
        ######################################################################
        ### Slave ############################################################
        ######################################################################
        
        from io import BytesIO
        import traceback
        
        try:
            # Get worker
            workerpickled = BytesIO(comm.bcast(obj=None, root=rank_master))
            worker = pickle.load(workerpickled)
            
            # Receive work and do it
            while True:
                inputdata = comm.recv(source=rank_master, tag=MPI.ANY_TAG, status=status)
                if status.Get_tag() == DIETAG:
                    break
                else:
                    result = worker(inputdata)
                    comm.send(obj=result, dest=rank_master)
        
        except Exception as err:
            comm.send(obj=None, dest=rank_master, tag=ERRORTAG)
            print("Slave with rank {} on {} terminated with an error:\n{}".format(rank, host, traceback.format_exc()))
