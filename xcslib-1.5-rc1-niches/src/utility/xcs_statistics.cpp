#include "xcs_statistics.h"
#include "xcs_utility.h"

//! reset all the collected statistics
void
xcs_statistics::reset()
{
	average_prediction = 0;
	average_fitness = 0;
	average_error = 0;
	average_actionset_size = 0;
	average_experience = 0;
	average_numerosity = 0;
	average_time_stamp = 0;
	average_no_updates = 0;	
	system_error = 0;

	no_cover = 0;
	no_ga = 0;
	no_condensation = 0;
	no_insert_classifiers = 0;
	no_subsumption = 0;
	no_crossover = 0;
	no_subsumptions_GA = 0;
	no_subsumptions_condensation = 0;
	no_subsumptions_in_A_GA = 0;
	no_subsumptions_in_A_condensation = 0;

	//! GA/Condensation/Selection timing
	ga_time = 0;
	ga_time_ticks = 0;

	selection_time = 0;
	selection_time_ticks = 0;

	condensation_time = 0;
	condensation_time_ticks = 0;

	selection_time_learning = 0;
	selection_time_ticks_learning = 0;
	no_selection_calls_learning = 0;	

	selection_time_condensation = 0;
	selection_time_ticks_condensation = 0;		
	no_selection_calls_condensation = 0;	


	//! used to track caching in the leaner approach
	cached_action_set_used = 0;
	new_action_set_computed = 0;

	cached_prediction_array_used = 0;
	new_prediction_array_computed = 0;
}

//! class constructor; it invokes the reset method \sa reset
xcs_statistics::xcs_statistics()
{
	reset();
}

ostream& 
operator<<(ostream& output, const xcs_statistics& stats)
{
	output << stats.average_prediction << "\t";
	output << stats.average_fitness << "\t";
	output << stats.average_error << "\t";
	output << stats.average_actionset_size << "\t";
	output << stats.average_experience << "\t";
	output << stats.average_numerosity << "\t";
	output << stats.average_time_stamp << "\t";
	output << stats.average_no_updates << "\t";
	output << stats.system_error << "\t";

	output << stats.no_ga << "\t";
	output << stats.no_cover << "\t";
	output << stats.no_subsumption << "\t";
	output << stats.no_crossover << "\t";

	return (output);
}

void xcs_statistics::pretty_print(ostream& output) 
const
{
	output << "# GA Activations " << no_ga << endl;
	output << "# Condensation Activations " << no_condensation << endl;
	output << "# Covering " << no_cover << endl;
	output << "# Subsumptions " << no_subsumption << endl;
	output << "# Subsumptions (GA) " << no_subsumptions_GA << endl;
	output << "# Subsumptions (condensation) " << no_subsumptions_condensation << endl;
	output << "# Subsumptions in [A] (GA) " << no_subsumptions_in_A_GA << endl;
	output << "# Subsumptions in [A] (condensation) " << no_subsumptions_in_A_condensation << endl;
	output << "# Crossovers " << no_crossover << endl;
	output << "# Insert " << no_insert_classifiers << endl;

	output << "# Cached Action Sets = " << cached_action_set_used << endl;
	output << "# Computed Action Sets = " << new_action_set_computed << endl;

	output << "# Cached Prediction Arrays = " << cached_prediction_array_used << endl;
	output << "# Computed Prediction Arrays = " << new_prediction_array_computed << endl;
	output << endl;

	output << "Timing" << endl;
	output << "GA Time (microseconds) = " << ga_time << endl;
	output << "GA Time Ticks (microseconds) = " << ga_time_ticks << endl;
	output << "Condensation Time (microseconds) = " << condensation_time << endl;
	output << "Condensation Time Ticks (microseconds) = " << condensation_time_ticks << endl;
	output << "Selection Time (microseconds) = " << selection_time << endl;
	output << "Selection Time Ticks (microseconds) = " << selection_time_ticks << endl;

	output << "Selection Time Learning (microseconds) = " << selection_time_learning << endl;
	output << "Selection Time Ticks Learning (microseconds) = " << selection_time_ticks_learning << endl;
	output << "# Selection Calls During Learning = " << no_selection_calls_learning << endl;

	output << "Selection Time Condensation (microseconds) = " << selection_time_condensation << endl;
	output << "Selection Time Ticks Condensation (microseconds) = " << selection_time_ticks_condensation << endl;
	output << "# Selection Calls During Condensation = " << no_selection_calls_condensation << endl;
}

istream& 
operator>>(istream& input, xcs_statistics& stats)
{
	input >> stats.average_prediction;
	input >> stats.average_fitness;
	input >> stats.average_error;
	input >> stats.average_actionset_size;
	input >> stats.average_experience;
	input >> stats.average_numerosity;
	input >> stats.average_time_stamp;
	input >> stats.average_no_updates;
	input >> stats.system_error;

	input >> stats.no_ga;
	input >> stats.no_cover;
	input >> stats.no_subsumption;
	return (input);
}

void xcs_statistics::save(const string& filename) const
{
    //! init the file for statistics
    ofstream STATS;

    STATS.open(filename, ios::out);
    if (!STATS.good())
    {
		xcs_utility::error("xcs_statistics", "save", "Cannot open stats file '"+string(filename), 1);
    }
	pretty_print(STATS);
    STATS.close();
}
