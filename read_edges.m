function A = read_edges(filename)
% READ_EDGES Read a graph from a list of edges
fid = fopen(filename,'rt');
n = fscanf(fid,'%d',1);
nz = fscanf(fid,'%d',1);
ai = zeros(nz,1);
aj = zeros(nz,1);
for i=1:nz
    ai(i) = fscanf(fid,'%d',1)+1;
    aj(i) = fscanf(fid,'%d',1)+1;
end
fclose(fid);
A = sparse(ai,aj,1,n,n);

