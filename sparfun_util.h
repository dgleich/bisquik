/**
 * @file sparfun_util.h
 * Utility functions for the sparse matrix function library
 */

/*
 * David F. Gleich
 * Copyright, 2008-2011
 * Developed while working at Stanford University, Microsoft Corporation,
 * the University of British Columbia, and Sandia National Labs.
 */

/** History
 *  :2008-09-01: Initial coding
 *  :2010-05-21: added randbyte/rand_uint
 */

double sf_time();

void sf_srand(unsigned long seed);
unsigned long sf_timeseed(void);
double sf_rand(double min=0.0, double max=1.0);
int sf_randint(int min, int max);
unsigned char sf_randbyte(void);
unsigned int sf_rand_uint(void);
unsigned int sf_rand_size(size_t maxval);

size_t sf_rand_distribution(size_t n, double* dist);

/** A small wall-clock timer class.
 *
 * Suggested usage: wall_timer t; 
 * my_computation_setup();
 * t.start(); my_computation(); dt = t.dt();
 */
#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
struct wall_timer {
    LARGE_INTEGER t0;
    LARGE_INTEGER freq;
    wall_timer() {QueryPerformanceFrequency(&freq); start();}
    void start() { QueryPerformanceCounter(&t0); }
    double dt() {
        LARGE_INTEGER t; 
        QueryPerformanceCounter(&t);
        return (double)(t.QuadPart - t0.QuadPart) / (double) freq.QuadPart;
    }
};
#else
#include <ctime>
#include <unistd.h>
#include <sys/time.h>
struct wall_timer {
    timeval t0;
    wall_timer() { start(); }
    void start() { gettimeofday(&t0, 0); }
    double dt() {
        timeval t; 
        gettimeofday(&t, 0);
        return (double)(t.tv_sec - t0.tv_sec) + (double)(
                    t.tv_usec - t0.tv_usec)/1000000.0;
    }
};
#endif
