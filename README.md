# mpimap for Python 3
A parallel map using mpi. The map is optimized for speed and usage within Jupyter notebooks. However, the code is still experimental and just tested on few systems.

It depends on the Python libraries

* mpi4py
* cloudpickle
* mmap

To make mpimap work more reliable, we have to slightly modify the cloudpickle library.
The cloudpickle library must not import uuid, this would create a new child process and could cause errors with mpi.
A quick and dirty workaround is to simply remove the `import uuid` statement from `cloudpickle/cloudpickle.py`.

## Examples

It is used the same as the `map` function of the Python multiprocessing library.
```python
from mpimap import Pool

def fct(idx):
  return idx

with Pool(hostfile="/PATH/TO/HOSTFILE") as p:
    results = p.map(fct, range(1000))
```

We can make use of the Python tqdm library, to show a progress bar.
```python
from mpimap import Pool
import tqdm # within a Jupyter notebook, call "from tqdm.notebook import tqdm" instead

def fct(idx):
  return idx

with Pool(hostfile="/PATH/TO/HOSTFILE") as p:
    results = list(tqdm(p.imap(fct, range(1000)), total=1000))
```
