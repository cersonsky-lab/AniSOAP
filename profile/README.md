# Code and Data used for Profiling AniSOAP Code

Please create data files and processing scripts here that help profile code.
We may not end up merging everything into the main branch especially if the input data or the outputs are large. 

Some random notes:

We should probably run these in CHTC for accurate results.
With the data that's here now, we can get the following results:
* How does how our code scales with number of frames?
* How does our code scale with lmax and nmax?


I am thinking to have one of each type of xyz file (1 species, 2 species, 3 species, 4 species), each with ~1000 frames.

Then, we can test the scaling -- the creation of the descriptor should scale as O(l*n^2*Z_{species}^2) according to dscribe (https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html, see the forloop). or O(l^2*n^2*Z_{species}^2) (https://pubs.aip.org/aip/jcp/article/154/11/114109/315400/Efficient-implementation-of-atom-density, see section f)

We want to test that scaling.

