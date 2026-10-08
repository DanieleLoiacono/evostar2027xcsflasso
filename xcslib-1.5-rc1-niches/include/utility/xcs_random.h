#ifndef __XCS_RANDOM__
#define __XCS_RANDOM__

#include <random>
#include "configuration_manager.h"

/*!
 * \class xcs_random
 *
 * \brief implements the random number generator for XCS
 *
 * \author Pier Luca Lanzi
 *
 * \version 0.01
 *
 * \date 2002/05/15
 *
 */


class xcs_random {
 private:
	//!  \var unsigned long seed to initialize the random number generator
	static unsigned long seed;

	static std::random_device rd;
	static std::mt19937_64 generator;
	static std::uniform_real_distribution<> uniform_distribution;	
	static std::normal_distribution<> normal_distribution;
	const static std::vector<std::string> supported_configuration_parameters;
 public:
	xcs_random(xcslib::configuration_manager& configuration);

	//! name of the class that implements the random number generator.
	static string class_name() { return string("xcs_random"); };
	
	//! tag used to access the configuration file
	static string tag_name() { return string("random"); };

	//! set the seed for random number generation through the configuration manager
	static unsigned long get_seed() { return seed; }
	static void set_seed(xcslib::configuration_manager&);
	static void print_parameters(ostream &OUTPUT);

	//! set the seed from random number generation
	static void set_seed(long new_seed);

	//! returns a random number in the real interval [0,1).
	static double random();

	//! returns a random number from a Gaussian distribution 
	static double nrandom();

	//! returns a random sign
	static int sign();

	/*! \brief returns an integer random number between 0 and limit-1.
	 *  \param limit the upper boundary for number generation
	 */
	static unsigned int dice(unsigned int limit);

	//! Saves the state of the random number generator to an output stream.
	/*! 
	 * It currently does not perform any action.
	 */
	static void save_state(ostream& output) {};

	//! Restores the state of the random number generator from an input stream.
	/*! 
	 * It currently does not perform any action.
	 */
	static void restore_state(istream& input) {};

	// needed for shuffling
	static std::mt19937_64 rng() {return generator;};
};
#endif
