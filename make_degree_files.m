%%
% This file is just a record of what comands I ran to produce degree files.

cd ..
cd My' Dropbox'\
dir
cd dev
ls
cd bis
cd bisquik\
ls
edit write_degree_file.m
load C:
load C:\cygwin\home\dfgleic\istanbul\04-02-graph-sigmamax\cs-stan.mat
A = A|A';
A = A - diag(diag(A));
degs_cs_stan_sym = sum(A,2);
load C:\cygwin\home\dfgleic\istanbul\04-02-graph-sigmamax\neuro-all.smat
addpath('C:\cygwin\home\dfgleic\istanbul\04-02-graph-sigmamax\');
A=readSMAT('C:\cygwin\home\dfgleic\istanbul\04-02-graph-sigmamax\neuro-all.smat'):
B=readSMAT('C:\cygwin\home\dfgleic\istanbul\04-02-graph-sigmamax\neuro-all.smat');
B=B|B';
degs_neuro_all = sum(A,2);
write_degree_file('neuro-all.degs',degs_neuro_all);
degs_neuro_all = sum(B,2);
write_degree_file('neuro-all.degs',degs_neuro_all);
write_degree_file('cs-stan-sym.degs',degs_cs_stan_sym);
load C:\cygwin\home\dfgleic\istanbul\04-02-graph-sigmamax\minnesota.smat
A=readSMAT('C:\cygwin\home\dfgleic\istanbul\04-02-graph-sigmamax\minnesota.smat');
A=A|A';
degs_mn = sum(A,2);
write_degree_file('minnesota.degs',degs_mn);
load C:\cygwin\home\dfgleic\istanbul\04-02-graph-sigmamax\tapir.mat
A = A|A';
degs_tapir = sum(A,2);
write_degree_file('tapir.degs',degs_tapir);