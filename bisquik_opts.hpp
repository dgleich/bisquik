/**
 * @file bisquik_opts.hpp
 * A header file for the command line options in bisquik.
 */
 
/**
 * @author David F. Gleich
 */

#include "argstream.h"

struct bisquik_options {
    bool verbose;
    bool stats;
    bool expprob;
    bool approxprob;
    std::vector<std::string> degfiles;
    
    bisquik_options() 
    : verbose(false)
    {}
    
    void print_options() {
        printf("verbose: %i\n", verbose);
        for (size_t i=0; i<degfiles.size(); ++i) {
            printf("degfile: %s\n", degfiles[i].c_str());
        }
    }
  
    void parse_command_line(int argc, char **argv) {
        argstream::argstream as(argc, argv);
        as >> argstream::option('v',"verbose",verbose,"Make the program chattier.");
        as >> argstream::option('s',"stats",verbose,"Collect output statistics.");
        as >> argstream::values<std::string>(std::back_inserter(degfiles),"degfiles");
        as >> argstream::help();
        as.defaultErrorHandling();
    }
};

bisquik_options opts;


