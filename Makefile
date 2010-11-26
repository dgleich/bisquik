
CC:=$(CXX)
CXXFLAGS += -std=c++0x -U__STRICT_ANSI__
all: bisquik

OBJS=bisquik.o sf_util.o

bisquik.o: bisquik_opts.hpp sparfun_util.h

bisquik: $(OBJS)  	
	g++ bisquik.o sf_util.o -o bisquik
	
