/**
 * @file sf_util.cc
 * Sparse matrix utility routines
 */

/*
 * David F. Gleich
 * Copyright, 2008-2011
 * Developed while working at Stanford University, Microsoft Corporation,
 * the University of British Columbia, and Sandia National Labs.
 */

/** History
 * 2008-09-01: Initial coding
 * 2010-09-13: Added functions from overlapping clusters code for
 *   more recent variate generator usage
 */

#if defined(_WIN32) || defined(_WIN64)
  #pragma warning(disable:4996)
  #include <random>
  #define tr1ns std::tr1
#elif defined __GNUC__
  #define GCC_VERSION (__GNUC__ * 10000 \
						   + __GNUC_MINOR__ * 100 \
						   + __GNUC_PATCHLEVEL__)
  #if GCC_VERSION < 40300
    #include <tr1/random>
    #define tr1ns std::tr1
    #define uniform_real_distribution uniform_real
    #define uniform_int_distribution uniform_int
  #else
    #include <random>
    #define tr1ns std
  #endif
#else
  #include <random>
  #define tr1ns std  
#endif  
    

#include <assert.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/timeb.h>
double sf_time()
{
#if defined(_WIN32) || defined(_WIN64)
  struct __timeb64 t; _ftime64(&t);
  return (t.time*1.0 + t.millitm/1000.0);
#else
  struct timeval t; gettimeofday(&t, 0);
  return (t.tv_sec*1.0 + t.tv_usec/1000000.0);
#endif
}

tr1ns::mt19937 sparfun_rand;

//typedef tr1ns::mt19937                                  generator_t;
//typedef tr1ns::uniform_real_distribution<double>        distribution_t;
//typedef tr1ns::variate_generator<generator_t, distribution_t> variate_t;
//variate_t sparfun_rand_unif(sparfun_rand, distribution_t(0.0, 1.0));

void sf_srand(unsigned long seed)
{
  sparfun_rand.seed(seed);
  //sparfun_rand_unif = variate_t(sparfun_rand, distribution_t(0.0, 1.0));
}


/** Return a seed based on the time. */
unsigned long sf_timeseed(void) 
{
    unsigned long seed = (unsigned long)sf_time();
    sf_srand(seed);
    return seed;
}

double sf_rand(double min0, double max0)
{
  tr1ns::uniform_real_distribution<double> dist(min0,max0);
  return dist(sparfun_rand);
}

unsigned char sf_randbyte(void)
{
  tr1ns::uniform_int_distribution<unsigned char> dist(0,255);
  return dist(sparfun_rand);   
}

unsigned int sf_rand_uint(void)
{
  tr1ns::uniform_int_distribution<unsigned int> 
            dist(0,std::numeric_limits<unsigned int>::max());
  return dist(sparfun_rand);
}

unsigned int sf_rand_size(size_t maxval)
{
    assert(maxval > 0);
    tr1ns::uniform_int_distribution<size_t> 
            dist(0,maxval-1);
  return dist(sparfun_rand);
}


int sf_randint(int min, int max)
{
  tr1ns::uniform_int_distribution<int> dist(min,max);
  return dist(sparfun_rand);
}

/** 
 * @param dist a cumulative probability distribution of size n
 *  where dist[n-1] = 1.0 and so dist[0] is the probability
 *  of sampling 0.
 */
size_t sf_rand_distribution(size_t n, double* dist)
{
    double rval = sf_rand(0.,1.);
    size_t i=0;
    // TODO add binary search here
    for (; i<n; i++) {
        if (dist[i]>rval) {
            break;
        }
    }
    
    return i;
}
