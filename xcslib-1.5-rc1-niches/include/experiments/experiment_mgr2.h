#include "xcs_definitions.h"
#include "xcs_random.h"
#include "timing.h"
#include "configuration_manager.h"
// #include "timer.h"

#ifndef __EXPERIMENT_MGR2__
#define __EXPERIMENT_MGR2__

class experiment_mgr2
{

public:
	//================================================================================
	//
	//
	//	PUBLIC METHODS
	//
	//
	//================================================================================
	
	//! name of the class that implements the experiment manager
	string class_name() const { return string("experiment_mgr"); };

	//! tag used to access the configuration file
	string tag_name() const { return string("experiments"); };

	//! class constructor; it reads the class parameters through the configuration manager
	experiment_mgr2(xcslib::configuration_manager &xcs_config, t_classifier_system *xcs, t_environment *environment, bool verbose=true);

	//! set the parameters from the configuration file
	void set_parameters(xcslib::configuration_manager & xcs_config);

	//! print the current parameter setting 
	void print_parameters(ostream&);

	//! perform the experiments
	void run();

	//! solve one episode
	void solve_episode(bool is_exploration_episode, bool is_condensation_active, unsigned long &no_steps, double &total_reward, double &system_error);

	// //! print the flags for save various experiment statistics
	// void print_save_options(ostream &output) const;

private:
	//================================================================================
	//
	//
	//	PRIVATE VARIABLES
	//
	//
	//================================================================================
	
	long	current_experiment;		//!< the experiment currently running
	long	first_experiment;		//!< number of the first experiment to run
	long	no_experiments;			//!< numbero of total experiments to run
	
	unsigned long	current_problem;		//!< the problem being currently executed
	unsigned long	first_learning_problem;		//!< first problem executed
	unsigned long	no_learning_problems;		//!< number of problems executed
	unsigned long	no_condensation_problems;	//!< number of problems executed in condensation
	unsigned long	no_test_problems;			//!< number of test problems executed at the end
	unsigned long	no_max_steps;				//!< maximum number of step per problem
	unsigned long	statistics_rolling_window; //!< window used to report statistics (0->statistics are printed each step)

	unsigned long	current_no_test_problems;	//!< number of test problems solved so far

	bool	flag_save_experiment_final_state;		//!< true if the state of the system will be saved at the end of the experiment
	long	save_experiment_interval;	//! the experiment status is saved every "save_interval" problems

	bool	flag_save_final_population;		//!< true if the state of the agent must be saved when an experiment ends
	long	save_population_interval;	//! the experiment status is saved every "save_interval" problems

	bool	flag_trace;					//!< true if the experiment outputs on the trace file
	bool	flag_test_environment;		//!< true if the system will be tested on the whole environment
	bool	flag_save_time_report;		//!< true if execution time is traced
	bool	flag_save_avf; 				//!< true if saves the action-value function

	string		extension;			//!< file extension for the experiment files
	
	unsigned long	save_stats_every;				//!< number of problems on which the average is computed and the statistics is reported

	unsigned long teletransportation_interval;	//! number of steps between teletransportation

	t_classifier_system *xcs;
	t_environment *environment;

	//================================================================================
	//
	//
	//	PRIVATE METHODS
	//
	//
	//================================================================================
	
	const static std::vector<std::string> configuration_parameters;

 private:

	//! save the agent state for experiment \emph expNo
	void save_population(unsigned long expNo, unsigned long problem_no=0) const;

	//! save time report
	void save_time_report(const xcslib::timing &experiment_timer, unsigned long experiment);

	//! save action value function 
	void save_avf(unsigned long expNo, unsigned long problem_no=0) const;

	//! sprintnf string lengths
	const static unsigned long max_filename_size = 1024;
	const static unsigned long max_system_command_size = 2048;

	void update_statistics(ostream &STATISTICS, const string& label, bool is_single_step,
							unsigned long current_experiment, unsigned long current_problem,
							unsigned long no_problem_steps, double reward_sum, unsigned long no_macro_classifier,
							double system_error) const;

	void update_rolling_statistics(ostream &STATISTICS, map<string, vector<double>> &rolling_statistics,
									bool is_single_step, unsigned long current_experiment,
									unsigned long current_problem, unsigned long first_learning_problem,
									unsigned long no_problem_steps, double reward_sum,
									unsigned long no_macroclassifiers, double system_error);

	void save_solution_statistics(ofstream &STATISTICS, map<string, vector<double>> rolling_solution_statistics,
									unsigned long current_experiment, unsigned long reported_problem,
									bool is_single_step) const;

};
#endif
