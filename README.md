# MpiProcessing - A Parallel Map Using MPI

[![Travis Build Status][travis-svg]][travis-link]
[![AppVeyor Build Status][appveyor-svg]][appveyor-link]

The Python 3 library *mpiprocessing* is a parallel map using mpi. The map is optimized for speed and usage within Jupyter notebooks. It is licensed under the [GPL v3][gpl-link].

It depends on the additional Python libraries

* mpi4py
* cloudpickle

## Examples

It is applied the same as the `map` and `imap` functions of the Python multiprocessing library.
```python
from mpiprocessing import Pool

def fct(idx):
  return idx

with Pool(hostfile="/PATH/TO/HOSTFILE") as p:
    results = p.map(fct, range(1000))
```

We can make use of the Python tqdm library to show a progress bar.
```python
from mpiprocessing import Pool
import tqdm # within a Jupyter notebook, call "from tqdm.notebook import tqdm" instead

def fct(idx):
  return idx

with Pool(hostfile="/PATH/TO/HOSTFILE") as p:
    results = list(tqdm(p.imap(fct, range(1000)), total=1000))
```

## Bugs

In case you encounter segmentation faults using mpiprocessing, you might slightly modify the cloudpickle library, to make mpiprocessing work more reliable. The cloudpickle library should not import uuid, this would create a new child process and could cause errors with mpi. A quick and dirty workaround is to remove the `import uuid` statement from `cloudpickle/cloudpickle.py`.

[travis-svg]: https://api.travis-ci.com/seweber/mpiprocessing.svg?branch=master
[travis-link]: https://travis-ci.com/seweber/mpiprocessing
[appveyor-svg]: https://ci.appveyor.com/api/projects/status/by19txrhh6fedjbe/branch/master?svg=true
[appveyor-link]: https://ci.appveyor.com/project/seweber/mpiprocessing/branch/master
[gpl-link]: https://www.gnu.org/licenses/gpl-3.0.html
