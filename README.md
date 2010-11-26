bisquik
======= 
bisquik randomly samples a graph with a prescribed degree distribution.

It implements the algorithm from Bayati, Kim, and Saberi, Algorithmica
[doi:...]


Usage
-----

    bisquik [options] degfile
    
    Generate a random sample of a graph with a prescribed degree 
    distribution.  The program reads the degrees in <degfile>
    and generates an asympotically uniform sample of a random graph
    with that same degree distribution.  If successful, the program
    outputs the edges of the graph to <degfile>.<k>.edges where
    <k> is the first integer such that <degfile>.<k>.edges does not
    exist.  
    
      -v, --verbose  make the program chattier
    
      -s, --stats  collect sampling statistics
        The statistics are written to <output>.stats
        
      -d PATH, --dir=PATH  change the output directory
        The default output name is <degfile>.<k>  Given PATH, 
        bisquik changes the path on <degfile> to PATH, 
        and searches for the first empty file with name 
        <PATH>/<degfile without path>.<k>.edges.  Changing the 
        output directory affects all other outputs as well.
        
      -o NAME, --output=NAME  the root output name
        The default output name is <degfile>.<k>  Given NAME, 
        bisquik searches for the first empty file with name 
        <NAME>.<k> just like the default behavior.  See --fixed 
        to avoid this behavior.
        
      -n COUNT, --samples=COUNT  produce COUNT samples
      
      -t COUNT, --trials=COUNT  perform COUNT trials for each sample 
        Each sample is not 
      
      -f NAME, --fixed=NAME  a fixed output name.
      
      --graphfile=NAME  an explicit graph filename for output
      
      --statsfile=NAME  an explicit statistics filename for output
          Using this option enables statistics collection
          
      -e, --expo  Sample edges with probability: exp(-di*dj/4m)*ri*rj
      
      -a, --approx  Sample edges with probability: (1-di*dj/4m)*ri*rj
          These are the probability used in the paper.
      
      --seed=<unsigned int>  If seed is 0, then the program is seeded
        based on the current time (the default).  
      
      -p <NVERTS>,<THETA>,<MAXDEG> --powerlaw=<NVERTS>,<THETA>,<MAXDEG>
        Ignore degfile and use a synthetically generated power-law
        degree distribution.
      
Degree File Format
------------------

The file format is textual and is a list of integers:
    
    File: Header\nDegreeList
    Header: <int:nverts>
    DegreeList: (<int:degree>\n)*nverts

For example, here is the degree sequence file for a triangle
with one extra node

    4
    2
    2
    3
    1


      
Acknowledgement
---------------

Uses the argparse.h library from ...



