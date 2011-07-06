


CC:=$(CXX)
CXXFLAGS += -std=c++0x -U__STRICT_ANSI__ -O3
all: bisquik

.PHONY: all clean perf

OBJS=bisquik.o sf_util.o

bisquik.o: bisquik_opts.hpp sparfun_util.h

bisquik: $(OBJS)  	
	g++ bisquik.o sf_util.o -o bisquik
	
perf: bisquik
	time -p ./bisquik as-22july06.smat.degs -t 50 --graphfile test.perf --seed 1
	rm -rf test.perf
	
clean:
	$(RM) -rf $(OBJS) bisquik

