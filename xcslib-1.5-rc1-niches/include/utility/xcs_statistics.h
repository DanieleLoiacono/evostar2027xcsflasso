/*!
 * \class xcs_statistics
 * \brief collects various XCS statistics
 */

#include <istream>
#include <ostream>
#include <fstream>

using namespace std;

#ifndef __XCS_STATISTICS__
#define __XCS_STATISTICS__

class xcs_statistics
{
	public:
		double	average_prediction;			//! average prediction in [P]
		double	average_fitness;			//! average fitness in [P]
		double	average_error;				//! average prediction error in [P]
		double	average_actionset_size;		//! average size of [A]
		double	average_experience;			//! average experience
		double	average_numerosity;			//! average numerosity
		double	average_time_stamp;			//! average time stampe
		double	average_no_updates;			//! average number of classifier updates
		double	system_error;				//! system error in [P]

		// unsigned long no_macroclassifiers;	//! number of macroclassifiers in [P]
		unsigned long no_ga;				//! number of GA activations
		unsigned long no_condensation;		//! number of condensation activations
		unsigned long no_cover;				//! number of covering activations 
		unsigned long no_subsumption;		//! number of subsumption activations
		unsigned long no_crossover; 		//! number of crossovers 
		unsigned long no_insert_classifiers;//! number of macro classifiers added to the populations

		unsigned long no_subsumptions_GA;
		unsigned long no_subsumptions_condensation;
		unsigned long no_subsumptions_in_A_GA;
		unsigned long no_subsumptions_in_A_condensation;

		double ga_time;
		unsigned long ga_time_ticks;

		double selection_time;
		unsigned long selection_time_ticks;		

		double selection_time_learning;
		unsigned long selection_time_ticks_learning;
		unsigned long no_selection_calls_learning;	

		double selection_time_condensation;
		unsigned long selection_time_ticks_condensation;		
		unsigned long no_selection_calls_condensation;	

		double condensation_time;
		unsigned long condensation_time_ticks;

		//! used to track caching in the leaner approach
		unsigned long cached_action_set_used;		//! # cached action set was used
		unsigned long new_action_set_computed;		//! # action set was computed

		unsigned long cached_prediction_array_used;		//! # cached prediction array was used
		unsigned long new_prediction_array_computed;	//! # prediction array was computed		

		//! class constructor; it invokes the reset method \sa reset
		xcs_statistics();

		//! reset all the collected statistics
		void reset();

		//! read the statistics from an output stream
		friend istream& operator>>(istream&, xcs_statistics&);

		//! write the statistics to an output stream
		friend ostream& operator<<(ostream&, const xcs_statistics&);

		//! pretty print the statistics
		void pretty_print(ostream&) const;

		//! save to file
		void save(const string& filename) const;

};	

#endif