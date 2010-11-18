
all: bisquik

bisquik: bisquik.cc sf_util.cc sparfun_util.h bisquik_opts.hpp
	g++ bisquik.cc sf_util.cc -o bisquik -std=c++0x -g
