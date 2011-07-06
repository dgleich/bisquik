/**
 * bisquik.cc
 * ==========
 * 
 * An implementation of the Bayati-Kim-Saberi algorithm to uniformly 
 * a graph with a given degree sequence.
 * 
 * Must be compiled with
 *   -std=c++0x
 * 
 * History
 * -------
 * :2010-08-31: Initial coding
 * :2011-06-06: Copyright assertion.
 */

/**
 * @author David F. Gleich
 */

/*
 * Copyright (2011) Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government 
 * retains certain rights in this software.
 */

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *       
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *      
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 */

#include <assert.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <queue>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iterator>

//#define USE_EDGE_HASHMAP
//note, using the edge_hashmap is NOT faster than linear search currently

#ifdef USE_EDGE_HASHMAP
#include <unordered_set>
#endif 

#include "sparfun_util.h"

#include "bisquik_opts.hpp"

// in a CSR format, EdgeType is the type of the pointer array 
// and vertex type is the type of the destination array.  
typedef int VertexType; // vertex type is a numeric type
typedef int EdgeType; // edge should index an array


struct graph {
    VertexType nverts;
    VertexType *dst;
    EdgeType *p;
    
    /** Output the graph in edges format
     * @return false if any write operation failed
     */
    bool write_as_edges(FILE* f) {
        size_t nedges = 0;
        int rval = 0;
        rval = fprintf(f, "%zu %zu\n", (size_t)nverts, (size_t)p[nverts]);
        if (rval < 0) { return false; }
        for (VertexType i=0; i<nverts; ++i) {
            for (EdgeType ei=p[i]; ei < p[i+1]; ++ei) {
                rval = fprintf(f, "%zu %zu\n", (size_t)i, (size_t)dst[ei]);
                if (rval < 0) { return false; }
            }
        }
        return true;
    }
    
    bool write_as_smat(FILE* f) {
        size_t nedges = 0;
        int rval = 0;
        rval = fprintf(f, "%zu %zu %zu\n", 
            (size_t)nverts, (size_t)nverts, (size_t)p[nverts]);
        if (rval < 0) { return false; }
        for (VertexType i=0; i<nverts; ++i) {
            for (EdgeType ei=p[i]; ei < p[i+1]; ++ei) {
                rval = fprintf(f, "%zu %zu 1\n", (size_t)i, (size_t)dst[ei]);
                if (rval < 0) { return false; }
            }
        }
        return true;
    }
        
};

void free_graph(graph g) {
	free(g.dst);
	free(g.p);
	g.dst = NULL;
	g.p = NULL;
	g.nverts = 0;
};

template <typename ValueType, typename SizeType>
SizeType sum_array(ValueType* array, SizeType nvals)
{
	SizeType rval = 0;
	for (SizeType i=0; i<nvals; ++i) {
		rval += array[i];
	}
	return rval;
}

graph alloc_graph_from_degrees(VertexType nverts, VertexType* degrees) 
{
	EdgeType sum_edges = sum_array(degrees, (EdgeType)nverts);
	graph rval;
	rval.nverts = nverts;
	rval.dst = (VertexType*)malloc(sizeof(VertexType)*(size_t)sum_edges);
	rval.p = (EdgeType*)malloc(sizeof(EdgeType)*(size_t)(nverts+1));
	return rval;
}
/** A sparse set class that permits O(1) membership, O(1) insert, O(1) delete.
 * This class is based on an array.  It presumes that there is a large
 * but finite item space, and that items have indices between 0 and n-1.
 * It allocates two arrays, which are inverse permutations of each other.
 * i.e. elements[indices[i]] = i.  The permutation has the property
 * that elements[0] ... elements[cur_element-1] are in the set.  Thus,
 * we can check membership in O(1) time with indices[i] <= cur_element.
 * Likewise, we can insert an element by permuting its position to the
 * cur_element position and incrementing cur_element.  Deleting
 * an element involves permuting its position to cur_element-1 and
 * decrementing cur_elements.
 * 
 * IndexType should be something like an int or size_t.
 */
template <class IndexType>
class sparse_set_array {
    public:
    std::vector<IndexType> indices;
    std::vector<IndexType> elements;
    size_t cur_elements;
    size_t max_elements;
    
    /** 
     * Construction time is O(n)
     */
    sparse_set_array(size_t _max_elements) 
     : indices(_max_elements), elements(_max_elements),
       cur_elements(0), max_elements(_max_elements)
    {
        for (IndexType i=0; i<_max_elements; ++i) {
            indices[i] = i;
            elements[i] = i;
        }
    }
    
    void insert(IndexType i) {
        assert(i < max_elements);
        if (indices[i] >= cur_elements) {
            assert(cur_elements < max_elements);
            if (indices[i] == cur_elements) {
                // we are done as it's already in position
            } else {
                // move the indices[i] into the cur_elements position
                // and then move elements[cur_elements] into 
                // the indices[i] position
                swap(i,indices[i],cur_elements);
            }
            cur_elements ++;
        } else {
            // it's already here!
        }
    }
    
    void remove(IndexType i) {
        assert(i < max_elements);
        if (indices[i] >= cur_elements) {
            // this element is already deleted
        } else {
            assert(cur_elements > 0);
            if (indices[i] == cur_elements-1) {
                // this node is the last one in the array, and so 
                // we don't have to move it!
            } else {
                // move element i into position cur_elements-1,
                // and move the element in position cur_elements-1 
                // to indices[i]
                swap(i,indices[i],cur_elements-1);
            }
            cur_elements --;
        }   
    }
    
    size_t size() { return cur_elements; }
    
    size_t count(IndexType i) {
        assert(i < max_elements);
        if (indices[i] < cur_elements) {
            return 1;
        } else {
            return 0;
        }
    }
    
private:
    void place(IndexType i, IndexType position) {
        indices[i] = position;
        elements[position] = i;
    }
    
    void swap(IndexType i, IndexType pi, IndexType j, IndexType pj) {
        if (pi != pj) {
            place(i,pj);
            place(j,pi);
        } else {
            assert(i == j);
        }
    }
    
    void swap(IndexType i, IndexType pi, IndexType pj) {
        swap(i,pi,elements[pj],pj);
    }
};


/** The growing fixed degree graph structure efficiently handles edge 
 * inserts and edge lookups when we have a prescribed maximum degree
 * for each vertex.
 * 
 * graph gdata = alloc_graph_from_degrees(nverts, degrees);
 * growing_fixed_degree_graph g(gdata, degree);
 * g.insert(i,j); (does not insert j,i)
 * g.is_edge(i,j);
 * 
 * TODO: Determine optimal strategy for checking edge existance
 */
struct growing_fixed_degree_graph {
	// the raw memory structure for the graph, usually 
	// this is a return value for another function
    graph g;
    
    // a pointer to the end of the list of valid edges at a vertex
    // the edges currently used at a vertex i are
    //   for (EdgeType ei=g.p[i]; ei<pend[i]; ++ei) {
    //     VertexType j=dst[ei]
    //   }
    std::vector<EdgeType> pend;

    
    // the vector of degrees for each vertex.  This is only used 
    // to make sure we don't violate the degree bound.
    const VertexType* degrees;
    
#ifdef USE_EDGE_HASHMAP    
    std::vector<std::unordered_set<VertexType> > edges;
#endif    
    
    /** 
     * @param g_ a graph with space allocated for a graph for the 
     * prescribed degrees
     * @param degrees_ the degree of each vertex in the graph
     */
    growing_fixed_degree_graph(graph& g_, const VertexType* degrees_)
    : g(g_), pend(g_.nverts), degrees(degrees_)
#ifdef USE_EDGE_HASHMAP
        , edges(g_.nverts)
#endif         
    {
        reset();
    }
    
    /** Return the current degree of a vertex */
    VertexType degree(VertexType i) {
		assert(i < g.nverts);
		assert(g.p[i] <= pend[i]);
		return pend[i]-g.p[i];
	}
	
	/** Return true if an edge exists
	 * Implements stupid linear search for now
	 * TODO Improve this call
	 * @param i the source of the edge
	 * @param j the destination of the edge
	 * @return true if the edge exists, false otherwise
	 */
	bool is_edge(VertexType i, VertexType j) {
        assert(i < g.nverts);
		assert(j < g.nverts);
#ifdef USE_EDGE_HASHMAP
        if (edges[i].count(j) != 0) {
            return true;
        } else {
            return false;
        }
#else        
		assert(g.p[i] <= pend[i]);
		for (EdgeType ei=g.p[i]; ei<pend[i]; ++ei) {
			if (g.dst[ei] == j) { return true; }
		}
		return false;
#endif        
	}
    
	/** Return true if an edge exists
     * This call is only guaranteed to work for symmetric
     * graphs and slightly optimizes the work in that case
	 * Implements stupid linear search for now
	 * TODO Improve this call
	 * @param i the source of the edge
	 * @param j the destination of the edge
	 * @return true if the edge exists, false otherwise
	 */
	bool is_symmetric_edge(VertexType i, VertexType j) {
        assert(i < g.nverts);
		assert(j < g.nverts);
        if (degree(i) < degree(j)) {
            return is_edge(i,j); 
        } else {
            return is_edge(j,i);
        }   
	}    
    
    void reset() {
        // initialize all the edge pointers to an empty graph
        EdgeType curedge = 0;
        for (VertexType i=0; i<g.nverts; i++) {
            g.p[i] = curedge;
            pend[i] = curedge;
            curedge += degrees[i];
        }
        g.p[g.nverts] = curedge;
#ifdef USE_EDGE_HASHMAP
        for (VertexType i=0; i<g.nverts; i++) {
            edges[i].clear();
        }
#endif        
    }
	
	/** Insert an edge into the graph
	 * This function does not check for multiple edges
	 * It runs in constant time.
	 */
	void insert(VertexType i, VertexType j) {
		assert(i < g.nverts);
		assert(j < g.nverts);
        if (degree(i) >= degrees[i]) {
            std::cout << degree(i) << " " << degrees[i] << std::endl;
            assert(degree(i) < degrees[i]);
        }
		g.dst[pend[i]] = j;
		pend[i] += 1;
#ifdef USE_EDGE_HASHMAP
        edges[i].insert(j);
#endif   
	}
};

/** 
 * This routine checks the Erdos-Gallai conditions on a degree
 * sequence to determine if the sequence is graphical.
 * A graphical sequence is a sequence of numbers that could
 * be the list of degrees for an unirected graph.
 * 
 * @param begin Must be a multi-pass iterator
 * @param end The end of the iterator
 * @return true if the sequence is graphical
 */
template <typename VertexSizeTypeItr>
bool is_graphical_sequence_bucket(VertexSizeTypeItr begin, VertexSizeTypeItr end)
{
    typedef typename std::iterator_traits<VertexSizeTypeItr>::value_type VertexSizeType;
    std::vector<VertexSizeType> residual_degrees(begin, end); 
    typedef typename std::vector<VertexSizeType>::const_iterator ResIter;
    // TODO optimize out this second pass by handling the construction
    // of the residual degrees vector ourselves (and optimizing for the
    // case when we can preallocate the vector)
    VertexSizeType max_degree = *(std::max_element(residual_degrees.begin(),
                                            residual_degrees.end()));
    VertexSizeType total_degree_mod_2 = 0;
    // initialize the bucket sort arrays
    // this list must go up to index max_degree+1, which is size max_degree+2
    std::vector<size_t> degree_pointer(max_degree+2,0);
    
    //printf("checking mod 2\n");
    for (ResIter ri=residual_degrees.begin(), rend=residual_degrees.end();
            ri != rend; ++ri) {
        
        VertexSizeType degree = *ri;
        
        total_degree_mod_2 += (degree % 2); // keep this from overflowing
        total_degree_mod_2 = total_degree_mod_2 % 2;
        
        degree_pointer[degree+1] += 1;
    }
    //printf("done checking mod 2\n");
    
    size_t cumsum = 0;
    for (VertexSizeType i = 0; i < max_degree + 2; ++i) {
        cumsum += degree_pointer[i];
        degree_pointer[i] = cumsum;
    }
    
    //printf("loading bucket sorted list\n");
    // this is the last iteration we need over the input
    for (; begin != end; ++begin) {
        VertexSizeType degree = *begin;
        //printf("Input: %i\n", degree);
        residual_degrees[degree_pointer[degree]] = degree;
        degree_pointer[degree]+=1;
    }
    //printf("finished bucket sorted list\n");
    
    //printf("shifted bucket sort pointers\n");
    // now shift the degrees down by one position
    for (VertexSizeType d = max_degree+1; d > 0; --d) {
        degree_pointer[d] = degree_pointer[d-1];
    }
    degree_pointer[0] = 0;
    //printf("done with bucket pointers\n");
    
    VertexSizeType last_degree = max_degree;
    
    // print out the bucket-sort
    /*for (size_t i = 0; i<max_degree+2; ++i) {
        printf("%zu ", degree_pointer[i]);
    }
    printf("\n");
    for (size_t i=0; i<degree_pointer[max_degree+1]; ++i) {
        printf("%i ", residual_degrees[i]);
    }
    printf("\n");*/
    
    // run the main part of the algorithm
    while (1) {
        
        // find the last degree with any remaining vertices
        while (last_degree > 0) {
            if (degree_pointer[last_degree+1] - degree_pointer[last_degree] == 0) {
                last_degree -= 1;
            } else { 
                break;
            }
        }
        
        if (last_degree == 0) {
            // we are done
            break;
        }
        
        //printf("found last_degree = %i\n", last_degree);
        
        size_t pos = degree_pointer[last_degree+1]-1;
        
        // get it's degree
        VertexSizeType degree = residual_degrees[pos];
        //printf("last_degree = %i, degree = %i\n", last_degree, degree);
        assert(last_degree == degree);
        // assign it 0 degree
        residual_degrees[pos] = 0;
        
        // remove this vertex
        degree_pointer[degree+1] -= 1;
        
        // determine how many vertices we need to search backwards
        // to find enough to satisfy this vertex
        VertexSizeType first_degree = degree;
        VertexSizeType rdegree = degree;
        while (rdegree > 0) {
            size_t ndegree = degree_pointer[first_degree+1] - 
                                degree_pointer[first_degree];
            if (ndegree >= rdegree) {
                // vertices with degree from first_degree to degree 
                // provide "degree" vertices
                break;
            } else {
                rdegree -= ndegree;
                first_degree -= 1; // we need more vertices
                if (first_degree == 0) {
                    // this indicates there are not 
                    // enough vertices left
                    return false;
                }
            }
            assert(rdegree > 0); // we should have broken above
        }
        // now decrement all the vertices in this space
        // we use the first "rdegree" vertices from first_degree.
        // and then all the vertices of larger degree
        for (VertexSizeType curdegree = first_degree; curdegree <= degree; 
            ++curdegree) 
        {
            if (curdegree > first_degree) {
                // set rdegree to the total number of vertices
                rdegree = degree_pointer[curdegree+1] - 
                            degree_pointer[curdegree];
            }
            // decrement the first rdegree vertices
            for (size_t v = 0; v < rdegree; ++v) {
                residual_degrees[degree_pointer[curdegree]] -= 1;
                degree_pointer[curdegree] += 1;
            }
        }   
        
    }
    
    return true;
}

/** 
 * This routine checks the Erdos-Galli conditions on a degree
 * sequence to determine if the sequence is graphical or not.
 */
int check_graphical_sequence(
    VertexType nverts,
    VertexType* degrees)
{
    std::priority_queue<VertexType > q;
    VertexType max_degree = 0;
    EdgeType total_degree = 0;
    
    for (VertexType i=0; i<nverts; ++i) {
        q.push(degrees[i]);
        total_degree += degrees[i];
        if (degrees[i]>max_degree) {
            max_degree = degrees[i];
        }
    }
    if (total_degree % 2 != 0) {
        return 0;
    }   
    if (max_degree > nverts) {
        return 0;
    }
    
    std::vector<VertexType> dlist(max_degree);
    size_t nsteps = 0;
    while (!q.empty()) {
        VertexType dn = q.top();
        q.pop();
        for (VertexType d = 0; d < dn; ++d) {
            if (q.empty()) {
                return false;
            }
            dlist[d] = q.top();
            q.pop();
        }
        for (VertexType d = 0; d < dn; ++d) {
            if (dlist[d] < 1) {
                // this means we failed
                // and would have pushed a negative degree to the list
                return 0;
            } else if (dlist[d] == 1) {
                // in this case, we just removed the vertex
                // because it's residual degree is 0.
            } else {
                q.push(dlist[d]-1);
            }
        }
        nsteps++;
    }
    
    return 1;
}    

/** Uniformly sampling graphs with a prescribed degree distribution
 * 
 * Given a graphical degree distribution, this class will output
 * a graph with that degree distribution which is uniformly chosen
 * from all graphs with that degree distribution.
 * 
 * See Bayati, Kim, Saberi, Algorithmica [doi:] for details about
 * the algorithm.
 * 
 * This implementation is designed to accommodate any distribution,
 * however, the theory only supports degree distributions with 
 * max(degree) \le sqrt(m), and the fast runtime bound only holds when
 * max(degree) \le sqrt(sqrt(m)).
 */
class bayati_kim_saberi_uniform_sampler {
    
public:
    // max_reject is the number of random mini_vertex samples
    // that are rejected before employing max_reject_strategy.
    size_t max_reject;
    
    // max_retries is the number of retries of the sampling process 
    size_t max_retries;
    
    enum {
        FAIL_ON_MAX_REJECT, // after rejecting, fail the sample
        SEARCH_ON_MAX_REJECT, // after rejecting, search for a valid edge
    } max_reject_strategy;
    
    enum {
		STANDARD_PROB, // use 1-di*dj/4m
		EXPO_PROB, // use e^{-di*dj/4m)
	} sampling_probability;
    
    // the number of vertices in the graph (just a dup of gdata.nverts)
    VertexType nverts;
    
    // the vector of degrees
    VertexType *degrees;
    VertexType max_degree;
    
    // the underlying graph data provided by the user
    // TODO is it possible for us to manage this memory too
    graph gdata;
    
    // the actual growing degree graph using memory from g
    growing_fixed_degree_graph g;
    
    // the vector of mini-vertices
    std::vector<VertexType> miniverts;
    // the vector of residual degrees
    std::vector<VertexType> r;
    
    // the numver of vertices with potential edges left (sum r>0)
    VertexType rverts;
    // the total number of edges in the graph (sum degrees)
    EdgeType nedges;
    // half of the edges (sum degrees/2)
    EdgeType half_edges;
    // current number of mini-vertices
    EdgeType curminis;
    
    // the vector of residual vertices
    // rvertset[0]...rvertset[rverts-1] is the current set of
    // residual vertices
    sparse_set_array<VertexType> rvertset;
    
    double edge_tolerance; // accept 
    
    double log_probability; // the log of the probability of the random sample
    
    /** 
     * Instiantiate a sampler for graphs with a prescribed degree sequence.
     * 
     * @param nverts the number of vertices in the graph
     * @param degrees the vector of degrees
     */
    bayati_kim_saberi_uniform_sampler(
		graph g_, VertexType *degrees_)
    : 
    max_reject(100), 
    max_retries(50), 
    max_reject_strategy(SEARCH_ON_MAX_REJECT),
	sampling_probability(STANDARD_PROB),
	nverts(g_.nverts), 
	degrees(degrees_), 
    max_degree(0),
	gdata(g_),
	g(gdata,degrees), 
	r(g_.nverts),
    rvertset(g_.nverts),
	edge_tolerance(0.),
	log_probability(1.)
    {
		nedges = sum_array(degrees, (EdgeType)nverts);
		half_edges = nedges/2;
	
		// allocate the minivertices
		miniverts.resize(nedges);
		// set the mini vertices
		EdgeType ei=0;
		for (VertexType i=0; i<nverts; i++) {
			for (VertexType j=0; j<degrees[i]; j++) {
				miniverts[ei] = i;
				ei++;
			}
			if (degrees[i]>max_degree) {
				max_degree = degrees[i];
			}
		}
		assert(miniverts.size() == (size_t)ei);
		
		// switch to expo prob if max degree is too large
		if ((double)max_degree > sqrt((double)nedges/2)) {
			sampling_probability = EXPO_PROB;
		}
	}
	
	/** Update information after adding an edge to the graph */
	void update_vertex_data_for_edge(VertexType i, EdgeType mvi) {
		assert(r[i]>=1);
		r[i] -= 1;
		if (r[i]==0) {
            //std::cout << "rverts = " << rverts << " curminis " << curminis << std::endl;
			assert(rverts>=1);
			rverts -= 1;
            rvertset.remove(i);
		}
	}
	
	void add_edge(VertexType i, VertexType j, EdgeType mvi, EdgeType mvj) 
	{	
		update_vertex_data_for_edge(i, mvi);
		update_vertex_data_for_edge(j, mvj);
        g.insert(i,j);
		g.insert(j,i);
		
		// move the mini vertices to end
		// first, make sure the minis vertex order is organized
		// this is only necessary to make the cases below easier
		// it might be able to be removed with more thought
		if (mvi>mvj) {
			// swap
			EdgeType tmp1 = mvi;
			mvi = mvj;
			mvj = tmp1;
			VertexType tmp2 = i;
			i = j;
			j = tmp2;
		}
		if (mvi >= curminis-2) {
			// don't need to do anything because they are already 
			// in the right position because mvi<mvj
		} else if (mvj >= curminis-2) {
			if (mvj==(curminis-1)) {
				miniverts[mvi] = miniverts[curminis-2];
				miniverts[curminis-2] = i;
			} else {
				// mvj = curminis-2
				miniverts[mvi] = miniverts[curminis-1];
				miniverts[curminis-1] = i;
			}
		} else {
			miniverts[mvi] = miniverts[curminis-2];
			miniverts[mvj] = miniverts[curminis-1];
			miniverts[curminis-2] = i;
			miniverts[curminis-1] = j;
		}
		// update current mini-vertex count
		curminis -= 2; 
	}
	
	void reset_data() {
		rverts = nverts; 
		// initialize to the degree vector
        std::copy(degrees, degrees+nverts, r.begin());
        
        // initialize the set of remaining vertices
        // and decrease rverts by any vertices of degree 0
        for (VertexType i=0; i<nverts; ++i) {
            if (r[i] == 0) {
                rverts -= 1;
                assert(rvertset.count(i) == 0);
            } else {
                rvertset.insert(i);
            }
        }
        curminis = (EdgeType)miniverts.size();
        g.reset();
	}
	
	double edge_probability(VertexType i, VertexType j) {
		double prob=0.;
		switch (sampling_probability) {
			case STANDARD_PROB:
				prob = 1.0-(double)degrees[i]*(double)degrees[j]/
                                (4.*(double)half_edges);
            break;
			case EXPO_PROB:
				prob = exp(-(double)degrees[i]*(double)degrees[j]/
                            (4.*(double)half_edges));
            break;
		}
        return prob;
	}
	
	bool accept_edge(VertexType i, VertexType j) 
	{
		if (i==j || g.is_symmetric_edge(i,j)) {
			return false;
		}
		double prob = edge_probability(i,j);
        assert(prob >= 0.);
        assert(prob <= 1.);
        return (sf_rand() < prob);
	}
    
    void print_edge(VertexType i, VertexType j, const char* prefix="") {
        std::cout << prefix << " " << i << " " << j << " with degrees " 
                  << "d[i]=" << g.degree(i) << ", r[i]=" << r[i] << " and "
                  << "d[j]=" << g.degree(j) << ", r[j]=" << r[j] << std::endl;
    }
    
    bool search_for_edge_in_minis(void) {
        // the mini vertices we chose
		EdgeType mi=0,mj=0; 
		// the vertices we chose
		VertexType vi=0,vj=0;
		// the total probability so far
		double total_prob=0.0;
        // if we found an edge
        bool found_edge = false;
		for (EdgeType mri=0; mri<curminis; ++mri) {
			for (EdgeType mrj=mri+1; mrj<curminis; ++mrj) {
				VertexType i = miniverts[mri];
				VertexType j = miniverts[mrj];
				if (i==j || g.is_symmetric_edge(i,j)) { continue; }
				double edge_prob = edge_probability(i,j);
				total_prob += edge_prob;
				if (sf_rand() <= edge_prob/total_prob) {
					// accept this edge
					mi = mri;
					mj = mrj;
					vi = i;
					vj = j;
					found_edge = true;
                    //print_edge(i,j);
                    assert(r[vi] > 0);
                    assert(r[vj] > 0);
				}
			}
		}
		if (found_edge) {
			// add the edge
			add_edge(vi,vj,mi,mj);
		}
		return found_edge;
    }
    
    EdgeType lookup_mini_vertex(VertexType v) {
        for (EdgeType i=0; i<curminis; ++i) {
            if (miniverts[i]==v) {
                return i;
            }
        }
        return miniverts.size()+1;
    }
    
    bool search_for_edge_in_verts(void) {
		// the vertices we chose
		VertexType vi=0,vj=0;
        
        // the total probability so far
		double total_prob=0.0;
        // if we found an edge
        bool found_edge = false;
        
        std::vector<int> neighbor_index(nverts,0);
        
        // now sample among these vertices
        for (VertexType li=0; li<rverts; ++li) {
            // TODO can optimize this search by removing 
            // vertices connected go li with r[i] = 0
            // index the neighbors for O(1) edge checking
            
            // lookup the vertex
            VertexType i = rvertset.elements[li];
            
            // index the neighbors for O(1) edge checking
            // TODO check this perf optimization
            /*for (EdgeType nzi = g.g.p[i]; nzi < g.pend[i]; ++nzi) {
                neighbor_index[g.g.dsts[nzi]] = 1;
            }*/
            
            for (VertexType lj=li+1; lj<rverts; ++lj) {
                
                VertexType j = rvertset.elements[lj];
                if (i==j || g.is_symmetric_edge(i,j)) {
                    continue;
                }
                // TODO: Check this performance optimization
                //if (i == j || neighbor_index[j]) {
                    //continue;
                //}
                assert(r[j] > 0);
                double edge_prob = (double)(r[i]*r[j])*edge_probability(i,j);
				total_prob += edge_prob;
				if (sf_rand() <= edge_prob/total_prob) {
					vi = i;
					vj = j;
					found_edge = true;
                    //print_edge(i,j);
                    assert(r[vi] > 0);
                    assert(r[vj] > 0);
				}
            }
            
            // clear the neighbor index
            /*for (EdgeType nzi = g.g.p[i]; nzi < g.pend[i]; ++nzi) {
                neighbor_index[g.g.dsts[nzi]] = 1;
            }*/
        }
		if (found_edge) {
            // find mini-verts
            EdgeType mi = lookup_mini_vertex(vi), mj = lookup_mini_vertex(vj);
            assert(mi < curminis);
            assert(mj < curminis);
			// add the edge
			add_edge(vi,vj,mi,mj);
		}
		return found_edge;
    }
        
    bool check_for_graphical_residuals(void) {
        assert(rverts == rvertset.size());
        
        // TODO figure out a way to avoid this copy
        std::vector<VertexType> rvertdegs(rverts);
        for (VertexType i=0; i<rverts; i++) {
            rvertdegs[i] = r[rvertset.elements[i]];
        }
        
        return is_graphical_sequence_bucket(rvertdegs.begin(),rvertdegs.end());
        
        /*if (check_graphical_sequence(rverts, &rvertdegs[0]) == 1) {
            return true;
        } else {
            return false;
        }*/
    }            
	
	bool search_for_edge_few_minis(void) {
        if (opts.verbose) {
            std::cout << "searching for edge with " << curminis <<
                " mini-vert and " << rverts << " vertices" << std::endl;
        }
		
        // check Erdos-Gallai conditions on the residuals
        if (!check_for_graphical_residuals()) {
            return false;
        }
        
        // this should always be faster now
        return (search_for_edge_in_verts());
        
        //if ((EdgeType)rverts + (EdgeType)sqrt(nverts) < curminis) {
            //return (search_for_edge_in_verts());
        //} else {
            //return search_for_edge_in_minis();
        //}
	}
	
	bool search_for_edge(void) {
		// TODO add another option to search when there
		// are still at lot of mini-vertices left.
        if (rverts < 10*(EdgeType)sqrt((double)max_degree)) {
            return search_for_edge_few_minis();
        } else {
            // at this point, search is going to be expensive
            // redo sampling because handling this case can be
            // expensive!
            for (VertexType t=0; t<rverts; ++t) {
                if (sample_edge()) {
                    return true;
                }
            }
            //okay sampling didn't help
            return search_for_edge_few_minis();
        }
            
	}
    
    /** Check status of minivertex structure 
     * This function is only used for debugging our datastructure
     * for consistency.
     */
    void check_miniverts(void) {
        std::vector<VertexType> cur_resid(nverts,0);
        for (EdgeType i = 0; i<curminis; ++i) {
            cur_resid[miniverts[i]] += 1;
        }
        VertexType check_rverts=0;
        for (VertexType i=0; i<nverts; ++i) {
            if (r[i] != 0) {
                check_rverts += 1;
            }
            if (cur_resid[i] != r[i]) {
                std::cout << "error on vertex " << i 
                    << " cur_resid="<<cur_resid[i] << " r=" << r[i] << std::endl;
                assert(cur_resid[i]==r[i]);
            }
        }
        if (check_rverts != rverts) {
            std::cout << "error with rverts " 
                      << "sum(r[i] != 0)="<<check_rverts 
                      << " but rverts="<<rverts 
                      <<std::endl;
            assert(check_rverts == rverts);
        }
        for (VertexType i=0; i<nverts; ++i) {
            if (r[i] + g.degree(i) != g.degrees[i]) {
                std::cout << "residual failed on vertex " << i
                    << " r=" << r[i] << " d=" << g.degree(i) 
                    << " target=" << g.degrees[i] << std::endl;
                assert(r[i] + g.degree(i) == g.degrees[i]);
            }
        }
    }
    
    bool sample_edge() {
        // sample an edge
        EdgeType mvi = (EdgeType)sf_rand_size(curminis);
        EdgeType mvj = (EdgeType)sf_rand_size(curminis);
        assert(mvi < curminis);
        assert(mvj < curminis);
        VertexType i = miniverts[mvi];
        VertexType j = miniverts[mvj];
        
        if (accept_edge(i,j)) {
            add_edge(i,j,mvi,mvj);
            return true;
        } else {
            return false;
        }
    }
            
	
	bool sample_one()
	{
		// reset the state
		reset_data();
        
        // the minivertices were allocated in the constructor
        // we do not need to reset them because the array ALWAYS
        // has the right counts of everything
        
        // the current number of edges added
        EdgeType curedges = 0;
        // the current number of minivertex sampling attempts
        size_t curtries = 0; // TODO see if there is a better type to use here
        // the number of vertices with edges remaining       
        EdgeType half_edges = miniverts.size()/2;
        
        while (curedges < half_edges) {
			if (curtries > max_reject) {
                if (opts.verbose) {
                    std::cout << "sampling failed on edge " << 
                        curedges << std::endl;
                }
				switch (max_reject_strategy) {
				case FAIL_ON_MAX_REJECT:
					return false;
					break; // out of case
				case SEARCH_ON_MAX_REJECT:
					// search for a valid edge
					if (search_for_edge()) {
                        curedges ++;
                        curtries = 0;
					} else {
                        return false;
                    }
                    break; // out of case
				}
			} else {
				if (sample_edge()) {
                    curedges ++;
                    curtries = 0;
                } else {
                    curtries++;
                }

			}
            // enabling this check makes things go really slow.
            // it is not needed unless you want to debug a change.
            //check_miniverts();
		}
        return true;
	}
    
    bool check_sample(void) {
        assert(rverts == 0);
        assert(curminis == 0);
        for (VertexType i=0; i<nverts; ++i) {
            assert(g.pend[i] == g.g.p[i+1]);
        }
        for (VertexType i=0; i<nverts; ++i) {
            assert(r[i] == 0);
        }
        return true;
    }
    
    /** Sample a graph with the prescribed degree sequence. 
     * 
     * This function is NOT thread safe.
     * 
     * @param g the output graph
     * @return true if the sample was successful or false if it
     *   could not complete
     */
    bool sample(void) 
    {
        for (size_t s = 0; s<max_retries; ++s) {
            if (sample_one()) {
                check_sample(); // should only be used while debugging
                return true;
            }
            if (opts.verbose) {
                std::cout << "failed sample " << s+1  << std::endl;
            }
        }
            
        return false;
    }
    
    struct sampling_statistics {
        std::vector< size_t > edge_queries;
        std::vector< std::pair<VertexType, VertexType> > edge_order;
        void alloc(EdgeType half_edges) {
            edge_order.resize(half_edges);
        }
    };
    sampling_statistics stats;
    
        
    /** Allocate extra memory to store statistics about the random sample
     * 
     * This information is useful for debugging what happens.
     */
    void collect_statistics() {
        
        
    }
    
};

void random_power_law_degrees(size_t n, double theta, size_t max_degree,
        VertexType* degrees)
{
    assert(theta>0);
    // first compute a distribution over degrees based on the powerlaw
    std::vector<double> dist(max_degree,0);
    double total = 0.;
    for (size_t i=0; i<max_degree; ++i) {
        dist[i] = (double)n*1./(pow(((double)i+1.),theta));
        total += dist[i];
    }
    // normalize to be a cumulative probability distribuiton
    double current_sum = 0.;
    for (size_t i=0; i<max_degree; ++i) {
        current_sum += dist[i]/total;
        dist[i] = current_sum;
    }
    
    // now sample n degrees
    EdgeType total_degrees = 0;
    for (size_t i=0; i<n; ++i) {
        degrees[i] = (VertexType)sf_rand_distribution(max_degree, &dist[0]) + 1;
        total_degrees += degrees[i];
    }
    if (total_degrees%2==1) {
        degrees[0] += 1;
    }
}

const size_t max_degree_seq_trials = 50;


/** Read a file of degrees.
 * The file format is textual and a list of integers:
 *   File: Header DegreeList
 * where 
 *   Header: <int:nverts>
 *   DegreeList: <int:degree>*nverts
 * and the degrees are in order.
 * 
 * @param filename the name of the degree file
 * @param degrees an output vector -- will be resized so that
 *   degrees.size() = nverts on output
 * @return false if reading fails for any reason
 */
bool load_degree_file(const char* filename, std::vector<VertexType>& degrees) 
{
    bool rval = true;
    using namespace std;
    ifstream t(filename);
    VertexType n = 0;
    t >> n; // read n
    if (t.fail()) {
        return false;
    }
    degrees.resize(n);
    for (VertexType i=0; i<n; ++i) {
        t >> degrees[i];
        if (t.fail()) {
            return false;
        }
    }
    t.close();
    return true;
}

/**
 * @return the number of successful samples.
 */
int generate_samples(const char* filename, std::vector<VertexType>& degrees)
{
    size_t n = degrees.size();
    graph g = alloc_graph_from_degrees((VertexType)n, &degrees[0]);
    bayati_kim_saberi_uniform_sampler generator(g, &degrees[0]);
    
    generator.max_retries = opts.trials;
    generator.max_reject = opts.max_reject;
    if (opts.stats) {
        generator.collect_statistics();
    }
    
    if (opts.expprob) {
        generator.sampling_probability = generator.EXPO_PROB;
    } 
    else if (opts.approxprob) {
        generator.sampling_probability = generator.STANDARD_PROB;
    }
    
    int nsamples = 0;
    for (int si=0; si<opts.samples; ++si) {
        if (generator.sample()) {
            std::cout << "successfully generated sample " << si+1 << std::endl;
            nsamples += 1;
        } else {
            std::cout << "failed generating sample " << si+1 << std::endl;
            continue;
        }
        
        // now figure out how to write out the file
        int index=1;
        FILE *gfile = opts.open_graph_file(filename, index);
        if (!gfile) {
            // output on STDOUT
            fprintf(stderr, "Error writing to file, outputing to stdout.");
            fprintf(stdout, "BEGIN_GRAPH");
            if (opts.format.compare("smat") == 0) {
                g.write_as_smat(stdout);
            } else if (opts.format.compare("edges") == 0) {
                g.write_as_edges(stdout);
            }
            fprintf(stdout, "END_GRAPH");
        } else {
            if (opts.format.compare("smat") == 0) {
                g.write_as_smat(gfile);
            } else if (opts.format.compare("edges") == 0) {
                g.write_as_edges(gfile);
            }
        }
        fclose(gfile);
        if (opts.stats) {
            FILE *sfile = opts.open_stats_file(filename, index);
            if (!sfile) {
                // output on stdout
            } else {
                // dump stats
            }
            fclose(sfile);
        }
    }
    free_graph(g);
    return nsamples;
}

int main(int argc, char **argv)
{
    if (!opts.parse_command_line(argc, argv)) {
        return (-1);
    }
    opts.print_options();
    if (!opts.validate()) {
        return (-1);
    }
    
    if (opts.seed == 0) {
        opts.seed = sf_timeseed();
    }
    sf_srand(opts.seed);
    
    int rval=0;
    
    if (opts.powerlaw.size() > 0) {
        // geneate the degree sequence
        std::vector<VertexType> degrees(0);
        
        size_t n=1000;
        double theta=2.;
        size_t max_degree=31;
        
        if (!opts.powerlaw_parameters(n, theta, max_degree)) {
            std::cerr << "error reading powerlaw parameters" << std::endl;
            return -1;
        }
        
        bool graphical=false;
        degrees.resize(n);
        std::cout << "generating powerlaw degree distribution with " 
                  << "n=" << n << " theta=" << theta 
                  << " max_degree=" << max_degree << std::endl;
        for (size_t trial=0; trial<max_degree_seq_trials; ++trial) {
            random_power_law_degrees(n, theta, max_degree, &degrees[0]);
            if (check_graphical_sequence(n, &degrees[0]) == 1) {
                graphical=true;
                break;
            } else {
                std::cout << "failed graphical test, trial " << trial+1 << std::endl;
            }
        }
        if (!graphical) {
            std::cerr << "failed to produce a graphical sequence" << std::endl;
            return (-1);
        } else {
            std::cout << "found graphical sequence " << std::endl;
        }
        
        const char *filename = NULL;
        if (opts.degfiles.size() > 0) {
            filename = opts.degfiles[0].c_str();
        } 
        int nsamples = generate_samples(filename, degrees);
        if (nsamples != opts.samples) {
            std::cout << "only generated " << nsamples << " of "
                << opts.samples << " graph samples" << std::endl;
            rval = opts.samples - nsamples;
        }
        
    } else {
        // all the degree files exist
        for (size_t fi=0; fi<opts.degfiles.size(); ++fi) {
            std::string filestr = opts.degfiles[fi];
            const char* filename = filestr.c_str();
            // read the degree file
            std::vector<VertexType> degrees;
            if (!load_degree_file(filename, degrees)) {
                std::cerr << "failed reading " << filename << std::endl;
                return (-1);
            } else {
                if (check_graphical_sequence(degrees.size(), &degrees[0]) != 1) {
                    std::cerr << "degree file " << filename 
                              << " does not contain a graphical sequence" 
                              << std::endl;
                    return (-1);
                }
            }
            std::cout << "using degree distribution from " << filename
                      << " with n=" << degrees.size() << std::endl;
                      
            int nsamples = generate_samples(filename, degrees);
            if (nsamples != opts.samples) {
                std::cout << "only generated " << nsamples <<
                    " of " << opts.samples << " requested samples "
                    << "for degree file " << filename << std::endl;
                rval += opts.samples - nsamples;
            }
                    
        }
    }
 
    return rval;
}
