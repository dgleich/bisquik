function write_degree_file(filename,degs)
% WRITE_DEGREE_FILE Output a degree file for the bisquik generator.
%
%   write_degree_file(filename,degrees) outputs the vector of degrees as a
%   degree file for the bisquik random graph generator.
%

% David F. Gleich
% 2010-11-25

fid = fopen(filename,'wt');
fprintf(fid,'%i\n', numel(degs));
for i=1:numel(degs)
    fprintf(fid,'%i\n', full(degs(i)));
end
fclose(fid);