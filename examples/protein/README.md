# Install
Prerequisites:
 - hdf5
 - Kokkos: https://github.com/kokkos/kokkos
   To configure Kokkos with CUDA support use the following configuration:
    `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/install/kokkos -DCMAKE_CXX_STANDARD=17 -D CMAKE_CXX_COMPILER=/path/to/kokkos_source/bin/nvcc_wrapper -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON /path/to/kokkos_source`
   Then install the library using `make install`

To install ArborX, you can use the following configuration:
`cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/ArborX/ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_PREFIX_PATH=/path/to/kokkos -DARBORX_ENABLE_EXAMPLES=ON /path/to/arborx_source`
Then compile the library using `make`. The binary can be found in
`./examples/protein`.

If Kokkos was configured with CUDA support, ArborX will automatically use CUDA.
The order of backend preferences is CUDA, OpenMP, and finally Serial.

# Run
To run the code simply use `./ArborX_Protein.exe`.
## Input
`ArborX_Protein.exe` requires three files: `prolint.h5`, `protein_indices.txt`,
and `not_protein_indices.txt`. `prolint.h5` is created using 
[mdconvert](https://www.mdtraj.org/1.9.7/mdconvert.html?highlight=mdconvert). 
`protein_indices.txt` and `not_protein_indices.txt` are produced using
`examples/python/convert_data.py`
## Output
Two different outputs are written:
 - arborx\_raw.txt: output directly the output from ArborX. For each frame, we
 output the `indices` and the `offsets`.  `indices` stores the indices of the objects 
 that satisfy the predicates. `offsets` stores the locations in the indices view that 
 start a predicate, that is, predicates(i) is satisfied by primitives(indices(j)) for 
 `offsets(i) <= j < offsets(i+1)`. Following the usual convention, `offsets(n) == indices.size()`, 
 where n is the number of queries that were performed and indices.size() is the total 
 number of collisions.
 - arborx.txt: output the indices for each atoms and for each frame. If there are no atom in the
 radius, we use -1.
