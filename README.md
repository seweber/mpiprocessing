# MpiMap - A Parallel Map Using MPI

[![Travis Build Status][travis-svg]][travis-link]
[![AppVeyor Build Status][appveyor-svg]][appveyor-link]

The Python 3 library *mpimap* is a parallel map using mpi. The map is optimized for speed and usage within Jupyter notebooks. It is licensed under the [GPL v3][gpl-link].

It depends on the additional Python libraries

* mpi4py
* cloudpickle
* numpy

## Examples

It is applied the same as the `map` and `imap` functions of the Python multiprocessing library.
```python
from mpimap import Pool

def fct(idx):
  return idx

with Pool(hostfile="/PATH/TO/HOSTFILE") as p:
    results = p.map(fct, range(1000))
```

We can make use of the Python tqdm library to show a progress bar.
```python
from mpimap import Pool
import tqdm # within a Jupyter notebook, call "from tqdm.notebook import tqdm" instead

def fct(idx):
  return idx

with Pool(hostfile="/PATH/TO/HOSTFILE") as p:
    results = list(tqdm(p.imap(fct, range(1000)), total=1000))
```

## Bugs

In case you encounter segmentation faults using mpimap, you might slightly modify the cloudpickle library, to make mpimap work more reliable. The cloudpickle library should not import uuid, this would create a new child process and could cause errors with mpi. A quick and dirty workaround is to remove the `import uuid` statement from `cloudpickle/cloudpickle.py`.

[travis-svg]: https://img.shields.io/travis/seweber/mpimap.svg?branch=master&style=flat&logo=travis
[travis-link]: https://travis-ci.com/seweber/mpimap
[appveyor-svg]: https://ci.appveyor.com/api/projects/status/gxrdxaaykt6wa20l/branch/master?svg=true
[appveyor-link]: https://ci.appveyor.com/project/seweber/mpimap/branch/master
[gpl-link]: https://www.gnu.org/licenses/gpl-3.0.html
