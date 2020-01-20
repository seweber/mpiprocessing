import os
import pickle
import mmap
import cloudpickle # Note that cloudpickle must not import uuid (would create a new child process and cause errors with mpi)
import subprocess
import struct
import socket
import tempfile

class Pool():
    def __init__(self, processes=None, hostfile=None, tmpdir=None, pythonenv=None, buffersize=5e8):
        self.processes = processes
        self.hostfile = hostfile
        self.tmpdir = tmpdir
        self.pythonenv = pythonenv
        self.buffersize = buffersize
        
    def __enter__(self):
        # Create temporary directory and paths
        self.tmp = tempfile.TemporaryDirectory(dir=self.tmpdir)
        self.path_lock = os.path.join(self.tmp.name, "lock")
        self.path_flags = os.path.join(self.tmp.name, "flags.comm")
        self.path_work = os.path.join(self.tmp.name, "work.comm")
        self.path_worker = os.path.join(self.tmp.name, "worker.comm")
        self.path_result = os.path.join(self.tmp.name, "result.comm")
 
        # Create new files
        with open(self.path_flags, "wb") as f:
            f.write(b"0")
        if not os.path.exists(self.path_work) or os.path.getsize(self.path_work) != self.buffersize:
            with open(self.path_work, "wb") as f:
                f.truncate(self.buffersize)
        if not os.path.exists(self.path_worker) or os.path.getsize(self.path_worker) != self.buffersize:
            with open(self.path_worker, "wb") as f:
                f.truncate(self.buffersize)
        if not os.path.exists(self.path_result) or os.path.getsize(self.path_result) != self.buffersize:
            with open(self.path_result, "wb") as f:
                f.truncate(self.buffersize)
                
        # Open memory maps
        self.f_flags = open(self.path_flags, "r+b")
        self.f_work = open(self.path_work, "r+b")
        self.f_worker = open(self.path_worker, "r+b")
        self.f_result = open(self.path_result, "rb")

        if os.name == 'nt':
            self.mm_flags = mmap.mmap(self.f_flags.fileno(), 0, access=mmap.ACCESS_READ|mmap.ACCESS_WRITE)
            self.mm_work = mmap.mmap(self.f_work.fileno(), 0, access=mmap.ACCESS_WRITE)
            self.mm_worker = mmap.mmap(self.f_worker.fileno(), 0, access=mmap.ACCESS_WRITE)
            self.mm_result = mmap.mmap(self.f_result.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            self.mm_flags = mmap.mmap(self.f_flags.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_READ|mmap.PROT_WRITE)
            self.mm_work = mmap.mmap(self.f_work.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_WRITE)
            self.mm_worker = mmap.mmap(self.f_worker.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_WRITE)
            self.mm_result = mmap.mmap(self.f_result.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_READ)
        
        # Setup environment
        my_env = os.environ.copy()
        if self.pythonenv is not None:
            my_env["VIRTUAL_ENV"] = self.pythonenv
            my_env["PATH"] = os.path.join(self.pythonenv,"bin") + ":" + my_env["PATH"]
        my_env["MKL_NUM_THREADS"] = "1"
        my_env["OPENBLAS_NUM_THREADS"] = "1"
        
        # Create mpi command
        master_host = socket.gethostname()
        path_balancer = os.path.join(os.path.dirname(os.path.realpath(__file__)), "balancer.py")
        cmd_additions = []
        if os.name == 'nt': # TODO use parameters
            cmd = ["mpiexec", "python", path_balancer, master_host, self.path_lock, self.path_flags, self.path_work, self.path_worker, self.path_result]
        else:
            if self.hostfile is not None:
                cmd_additions += ["--hostfile",  self.hostfile]
            if self.processes is not None:
                cmd_additions += ["-np", str(self.processes)]
            if "VIRTUAL_ENV" in my_env:
                cmd_additions += ["-x", "VIRTUAL_ENV"]
            if "PATH" in my_env:
                cmd_additions += ["-x", "PATH"]
            cmd = ["mpiexec",
                #"--fwd-mpirun-port",
                #"--mca", "pmix_base_async_modex", "1",
                #"--mca", "btl", "^openib", 
                #"--mca", "btl_base_warn_component_unused", "0",
                "--mca", "orte_base_help_aggregate", "0",
                "--mca", "mpi_warn_on_fork", "1"
                ] + cmd_additions + ["-x", "MKL_NUM_THREADS", "-x", "OPENBLAS_NUM_THREADS",
                "python3", path_balancer, master_host, self.path_lock, self.path_flags, self.path_work, self.path_worker, self.path_result]

        # Start MPI, communicate with the master process using mmap
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=my_env)
        return self
    
    def __exit__(self, type, value, traceback):
        # Wait until mpiexec finished
        print(self.proc.communicate()[0].decode('utf-8'), end="")
                
        # Close memory maps
        self.mm_flags.close()
        self.f_flags.close()
        self.mm_work.close()
        self.f_work.close()
        self.mm_worker.close()
        self.f_worker.close()
        self.mm_result.close()
        self.f_result.close()
        
        # Delete temperoary directory
        self.tmp.cleanup()
    
    def _setflag(self, flag):
        self.mm_flags.seek(0)
        self.mm_flags.write(str(flag).encode())
    
    def _waitflag(self, flag):
        while True:
            self.mm_flags.seek(0)
            string = self.mm_flags.read()
            if int(string.decode()) >= flag or self.proc.poll() is not None: break
    
    def _isflag(self, flag):
        self.mm_flags.seek(0)
        string = self.mm_flags.read()
        return int(string.decode()) == flag

    def map(self, worker, params):        
        # Set worker
        self.mm_worker.seek(0)
        dump = cloudpickle.dumps(worker)
        self.mm_worker.write(struct.pack('?', 0)) # Do not set is_imap
        self.mm_worker.write(struct.pack('L', len(dump)))
        self.mm_worker.write(dump)
        self._setflag(1)

        # Set work
        self.mm_work.seek(0)
        pickle.dump(params, self.mm_work)
        self._setflag(2)
        
        #for line in iter(self.proc.stdout.readline, b""):
        #    if line == b"": break
        #    print(line.decode('utf-8'), end="")

        # Receive results
        self._waitflag(3)
        self.mm_result.seek(0)
        results = pickle.load(self.mm_result)
        
        return results
    
    def imap(self, worker, params):        
        # Set worker
        self.mm_worker.seek(0)
        dump = cloudpickle.dumps(worker)
        self.mm_worker.write(struct.pack('?', 1)) # Set type is_imap
        self.mm_worker.write(struct.pack('L', len(dump)))
        self.mm_worker.write(dump)
        self._setflag(1)

        # Set work
        self.mm_work.seek(0)
        pickle.dump(params, self.mm_work)
        self._setflag(2)

        # Load and yield part of results
        while True:
            if self._isflag(3):
                self.mm_result.seek(0)
                results = pickle.load(self.mm_result)
                self._setflag(4)
                for r in results:
                    yield r
            elif self._isflag(5) or self.proc.poll() is not None:
                break
    
