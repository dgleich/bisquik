/**
 * @file bisquik_opts.hpp
 * A header file for the command line options in bisquik.
 */
 
/**
 * @author David F. Gleich
 */

/*
 * Copyright 2011 Sandia Corporation. 
 * Under the terms of Contract DE-AC04-94AL85000, 
 * there is a non-exclusive license for use of 
 * this work by or on behalf of the U.S. Government.
 * Export of this program may require a license from
 * the United States Government.
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

#include "argstream.h"

#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Test if a file exists.
 *
 * @filename the name of the file
 * @return true if the file exists, false otherwise
 */
bool file_exists(const char* filename)
{
    using namespace std;
    ifstream t(filename);
    t.close();
    if (t.fail()) { return false; }
    else { return true; }
}

/** Split a string.  
 * Taken from 
 * http://stackoverflow.com/questions/236129/c-how-to-split-a-string
 */
std::vector<std::string>& split(
    const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

template <typename NumberType>
NumberType string_to_number(std::string str, bool& rval) {
    std::istringstream s(str);
    NumberType n;
    s >> n;
    if (s.fail()) {
        rval = false;
    } else {
        rval = true;
    }
    return n;
}

/** This function tests if it can open a filename for writing.
 * The algorithm is:
 *   i) try and open it with
 *     open(filename,O_APPEND|O_WRONLY);
 *   ii) if that succeeds, the file if writable, otherwise
 *     it may not exist
 */
bool is_filename_writable(const char* filename) {
    int fd=open(filename, O_WRONLY|O_CREAT|O_EXCL,
        S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd==-1) {
        int err = errno;
        if (err == EEXIST) {
            // check if we can open it for writing
            fd=open(filename, O_WRONLY|O_APPEND);
            if (fd < 0) {
                // we cannot write this file
                return false;
            } else {
                // we can write this file
                close(fd);
                return true;
            }
        } else {
            return false;
        }
    } else {
        // the file didn't exist
        // close and unlink the one we just created
        close(fd);
        unlink(filename);
        return true;
    }
}

/** Return a file pointer, but only create a new file. 
 * @return NULL if the file already exists or cannot be opened,
 *   otherwise, return a file pointer to a new file 
 */
FILE* open_if_does_not_exist(const char* filename) { 
    int fd=open(filename, O_WRONLY|O_CREAT|O_EXCL,
        S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd==-1) {
        return NULL;
    } else {
        FILE* f = fdopen(fd, "wt");
        return f;
    }
}




struct bisquik_options {
    bool verbose;
    bool stats;
    bool expprob;
    bool approxprob;
    
    std::string output;
    unsigned int samples;
    unsigned int trials;
    unsigned int max_reject;
    std::string output_dir;
    std::string output_fixed;
    std::string graph_filename;
    std::string stats_filename;
    std::string format;
    
    unsigned int seed;
    
    std::string powerlaw;
    
    std::vector<std::string> degfiles;
    
    bisquik_options() 
    : verbose(false), stats(false), expprob(true), approxprob(false),
        samples(1), trials(50), max_reject(100), format("edges"), seed(0)
    {}
    
    void print_options() {
        printf("verbose: %i\n", verbose);
        printf("stats: %i\n", stats);
        printf("samples: %u\n", samples);
        printf("trials: %u\n", trials);
        printf("rejects: %u\n", max_reject);
        printf("seed: %u\n", seed);
        printf("output: %s\n", output.c_str());
        printf("output_dir: %s\n", output_dir.c_str());
        printf("graph_filename: %s\n", graph_filename.c_str());
        printf("stats_filename: %s\n", stats_filename.c_str());
        printf("powerlaw: %s\n", powerlaw.c_str());
        printf("format: %s\n", format.c_str());
        
        for (size_t i=0; i<degfiles.size(); ++i) {
            printf("degfile: %s\n", degfiles[i].c_str());
        }
    }
    
    void usage() {
        fprintf(stderr,"usage - see README.md\n");
        // TODO finish writing usage
    }
  
    bool parse_command_line(int argc, char **argv) {
        argstream::argstream as(argc, argv);
        as >> argstream::option('v',"verbose",verbose,
                "Make the program chattier.");
        as >> argstream::option('s',"stats",stats,
                "Collect output statistics.");
        as >> argstream::parameter('o',"output",output,
                "Output basename.",false);
        as >> argstream::parameter('d',"dir",output_dir,
                "Output directory.",false);
        as >> argstream::parameter('n',"samples",samples,
                "Number of graph samples.",false);
        as >> argstream::parameter('t',"trials",trials,
                "Trials per sample.",false);
        as >> argstream::parameter('r',"rejections",max_reject,
                "Rejection steps before searching.",false);
        as >> argstream::parameter('f',"fixed",output_fixed,
                "A fixed output name.",false);
        as >> argstream::parameter("graphfile",graph_filename,
                "A fixed graph file output name.", false);
        as >> argstream::parameter("statsfile",stats_filename,
                "A fixed stats file output name.", false);
        as >> argstream::parameter("seed",seed,
                "A random seed.", false);
        as >> argstream::option('e',"expo",expprob,
                "Select edges with exponential probability");
        as >> argstream::option('a',"approx",approxprob,
                "Select edges with approximate exponential probability");
        as >> argstream::parameter('p',"powerlaw",powerlaw,
                "Sample from a power law to produce the degree sequence.",
                false);
        as >> argstream::parameter("format",format,
                "The graph output format: \'edges\' or \'smat\'",
                false);
        
        as >> argstream::values<std::string>(std::back_inserter(degfiles),"degfiles");
        as >> argstream::help();
        
        // handle errors ourself
        // as.defaultErrorHandling();
        if (as.helpRequested()) {
            usage();
            return false;
        } 
        
        // check for unused options
        std::cerr << "here" << std::endl;
        
        if (!as.isOk()) {
            std::cerr << as.errorLog() << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool _fixed_output() {
        return (graph_filename.size() > 0 || stats_filename.size() > 0 || 
            output_fixed.size() > 0 || output.size());
    }
    
    bool validate() {
        // pick filenames
        if ( _fixed_output() ) {
            // they specified a precise filename.  that means
            // one or zero degfiles
            if (degfiles.size() > 1) {
                std::cerr 
                 << "cannot use multiple degree files with fixed output names."
                 << std::endl;
                return false;
            } else if (samples > 1) {
                std::cerr
                 << "cannot use multiple samples with fixed output names."
                 << std::endl;
                return false;
            }
            
            if (graph_filename.size() > 0 && stats &&
                stats_filename.size() == 0) {
                std::cerr
                 << "when graphfile is specified and stats are enabled"
                 << std::endl
                 << "then statsfile must also be specified."
                 << std::endl;
                return false;
            }
            
        } else if (powerlaw.size() > 0) {
            if (degfiles.size() > 1) {
                std::cerr 
                 << "when powerlaw is specified, then degfile and --output"
                 << std::endl 
                 << "have the same meaning, and only one degfile is allowed."
                 << std::endl;
                return false;
            }
            if (degfiles.size() == 0 && !_fixed_output()) {
                std::cerr
                 << "when powerlaw is specified, then degfile or --output"
                 << std::endl
                 << "must be specified to pick the output name."
                 << std::endl;
                return false;
            }
        } else {
            // we need to read degree files.
            for (size_t fi = 0; fi < degfiles.size(); ++fi) {
                if (!file_exists(degfiles[fi].c_str())) {
                    std::cerr
                     << "degree file " << degfiles[fi] << " does not exist."
                     << std::endl;
                    return false;
                }
            }
        }
        
        // TODO check format.
        
        return true;
    }
    
    // handle the output dir option
    std::string _handle_path(std::string filename, bool file=false) {
        if (output_dir.size() == 0) {
            return filename; // do nothing if output_dir isn't specified.
        } else {
            // if this was not a filename listed, then 
            // it should be relative to the path
            if (file == false) {
                return output_dir + "/" + filename;
            }
            
            // split the filename
            size_t idx = filename.rfind("/");
            if (idx == std::string::npos) {
                // no path separator
                return output_dir + "/" + filename;
            } else {
                // separate the path
                std::string file_and_path_sep = filename.substr(idx);
                return output_dir + file_and_path_sep;
            }
        }
    }
    
    static const int max_index = 9999;
    
    FILE* _find_filename_combo(std::string base, int& index, const char* ext) {
        while (index < max_index) {
            std::ostringstream os;
            os << base << "." << index << ext;
            FILE* fp = open_if_does_not_exist(os.str().c_str());
            if (fp) {
                return fp;
            } else {
                index += 1;
            }
        }
        return NULL;
    }
    
    FILE* open_graph_file(const char* filename, int& index) {
        if (graph_filename.size() > 0) {
            return fopen(_handle_path(graph_filename).c_str(), "wt");
        }
        
        std::string ext = "." + format;
        
        if (output_fixed.size() > 0) {
            return fopen(_handle_path(output_fixed + ext).c_str(), "wt");
        }
        
        if (output.size() > 0) {
            index = 1;
            FILE* f = _find_filename_combo(_handle_path(output), index, ext.c_str());
            return f;
        } 
        
        if (filename) {
            index = 1;
            FILE* f = _find_filename_combo(_handle_path(filename), index, ext.c_str());
            return f;
        }
        
        assert(false);
        
        return (NULL);
    }
    
    FILE* open_stats_file(const char* filename, int index) {
        if (stats_filename.size() > 0) {
            return fopen(_handle_path(graph_filename).c_str(), "wt");
        }
        
        if (output_fixed.size() > 0) {
            return fopen(_handle_path(output_fixed + ".stats").c_str(), "wt");
        }
        
        if (output.size() > 0) {
            std::ostringstream os;
            os << output << index << ".stats";
            FILE* f = fopen(os.str().c_str(),"wt");
            return f;
        }
        
        if (filename) {
            std::ostringstream os;
            os << filename << "." << index << ".stats";
            FILE* f = fopen(os.str().c_str(),"wt");
            return f;
        }
        
        assert (false);
        
        return (NULL);
        
    }
    
    /**
     * Powerlaw parameters are:
     *   n,theta,maxdeg
     * where _ yields a default parameter.
     */
    bool powerlaw_parameters(size_t& n, double& theta, size_t& maxdeg) {
        std::vector<std::string> parts;
        split(powerlaw, ',', parts);
        if (parts.size() == 0 || parts.size() > 3) {
            std::cerr << "powerlaw parameters \'" << powerlaw << 
                "\' are not valid" << std::endl;
            return false;
        }
        // set defaults now
        n = 1024;
        theta = 2.2;
        maxdeg = 10;
        // handle n
        if (parts.size() >= 1) {
            std::string p = parts[0];
            if (p.compare("_") != 0) {
                // parse an integer
                bool rval = false;
                n = string_to_number<size_t>(p,rval);
                if (!rval) {
                    std::cerr << "the powerlaw size parameter \'" <<
                        p << "\' is not a valid integer > 0" << std::endl;
                    return false;
                }
            }
        }
        
        if (parts.size() >= 2) {
            std::string p = parts[1];
            if (p.compare("_") != 0) {
                // parse a double for theta
                bool rval = false;
                theta = string_to_number<double>(p, rval);
                if (!rval || theta<=0) {
                    std::cerr << "the powerlaw exponent parameter \'" <<
                        p << "\' is not a valid double > 0" << std::endl;
                    return false;
                }
            }
        }
        
        if (parts.size() >= 3) {
            std::string p = parts[2];
            if (p.compare("_") != 0) {
                // parse a double for theta
                bool rval = false;
                maxdeg = string_to_number<size_t>(p, rval);
                if (!rval) {
                    std::cerr << "the powerlaw max_degree parameter \'" <<
                        p << "\' is not a valid integer > 0" << std::endl;
                    return false;
                }
            }
        }
            
                
        return true;
    }
};

bisquik_options opts;


