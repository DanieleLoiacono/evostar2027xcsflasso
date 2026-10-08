#include "xcs_utility.h"
#include "xcs_statistics.h"
#include "timer.h"
#include "high_resolution_timer.h"
#include "experiment_mgr2.h"
#include "xcs_definitions.h"

/*!
 * \file experiment_mgr.cpp
 *
 * \brief implements the methods for the experiment manager 
 *
 */

const std::vector<std::string> experiment_mgr2::configuration_parameters = {"first experiment","number of experiments","first problem","number of learning problems","number of condensation problems","number of test problems","maximum number of steps","save final population","save population every","save experiment final state","save experiment state every","save problem execution trace","teletransportation interval","evaluate solution","save execution time report", "save action-value function", "statistics rolling window size"};

experiment_mgr2::experiment_mgr2(xcslib::configuration_manager &xcs_config, t_classifier_system *xcs, t_environment *environment, bool verbose)
{
	this->xcs = xcs;
	this->environment = environment;

	extension = xcs_config.extension();

	if (!xcs_config.exist(tag_name()))
	{
		xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
	}

	xcs_config.check_parameters(tag_name(),configuration_parameters);
    set_parameters(xcs_config);
}

void
experiment_mgr2::run()
{
	char			system_command[max_system_command_size];	//! string used to perform system calls.
	char			fn_statistics[max_filename_size];		//! filename statistics file
	char 			fn_trace[max_filename_size];			//! filename trace file
	ofstream		STATISTICS;					//! files that contains the whole experiment statitics. One line for each problem performed.
	ofstream		TRACE;						//! files that contains the trace information about the experiment

	double			reward_sum = 0;				//! sum of rewards gained while solving the problem
	unsigned long	problem_steps = 0;			//! number of steps needed to solve the problem

	
	//! timing problem execution
	xcslib::high_resolution_timer timer_problem(xcslib::high_resolution_timer::resolution::microseconds);
	
	xcslib::timing experiment_timer;

	bool			is_single_step = environment->is_single_step();

	experiment_timer.clear();
	
	//! performs all the experiments, one by one.
	for(current_experiment=first_experiment;
		current_experiment < (first_experiment+no_experiments); 
		current_experiment++)
	{
		t_state initial_state; 

		//! init XCS for the current experiment
		xcs->begin_experiment();
		
		//! init the environment for the current experiment
		environment->begin_experiment();
		
		//! the first problem is always solved in exploration
		bool flag_exploration = true;

        //! true when condensation should be used
        bool is_condensation_active = false;

		//! init the file for statistics
		snprintf(fn_statistics, max_filename_size, "statistics.%s-%04ld", extension.c_str(), current_experiment);
		
		//! init the statistics file for a new experiment
		STATISTICS.open(fn_statistics);
		if (!STATISTICS.good())
		{
			xcs_utility::error(class_name(),"perform_experiments","Statistics file '"+string(fn_statistics)+"' not open",1);
		}

		if (flag_trace)
		{
			snprintf(fn_trace, max_filename_size, "trace.%s-%04ld", extension.c_str(), current_experiment);
			TRACE.open(fn_trace);
			if (!TRACE.good())
			{
				xcs_utility::error(class_name(),"StartSession","Could not open trace file "+string(fn_trace),1);
			}
		}		

		//! init the timing for the experiment
		experiment_timer.clear();

		//! in single step problem like regression ones system_error = |reward-expected_payoff|
		double system_error; 

		map<string,vector<double>> rolling_statistics {
			{"Steps", vector<double>(statistics_rolling_window)},
			{"Reward", vector<double>(statistics_rolling_window)},
			{"Macroclassifier", vector<double>(statistics_rolling_window)},
			{"System Error", vector<double>(statistics_rolling_window)},
		};

		//--------------------------------------------------------------------------------
		// Solving Learning and Condensation Problems 
		//--------------------------------------------------------------------------------

		for(current_problem=first_learning_problem;
			current_problem<first_learning_problem+2*(no_learning_problems+no_condensation_problems); 
			current_problem++)
		{
			//! if needed save information in the trace file
			if (flag_trace)
			{
				TRACE << current_experiment << "\t" << current_problem;
			}
			
			//! is it the time to start condensation?
			if ((no_condensation_problems>0) && 
				(current_problem==first_learning_problem+2*no_learning_problems))
			{	
                //! turn on condensation
				is_condensation_active = true;
			}

			//! start timer for problem
			timer_problem.start_timer();

            //! init XCS for the problem starts (does not do anything)
            xcs->begin_problem();

            //! init the environment for the current problem
            environment->begin_problem();

			solve_episode(flag_exploration, is_condensation_active, problem_steps, reward_sum, system_error);

			//! stops the timer for the problem
			timer_problem.stop_timer();

			//! update the timing statistics
			experiment_timer.update(timer_problem.elapsed(),flag_exploration,is_condensation_active, problem_steps);

			//! problem trace information is saved
			/*! by default the statistics file contain (for each line)
			 *  - experiment number
			 *  - problem number
			 *  - trace information from XCS (usually null)
			 *  - trace information from the environment
			 *  - "Learning/Testing" whether the problem has been solved in learning or testing mode
			 */

			if (flag_trace)
			{
				TRACE << "\t" << xcs->trace() << "\t" << environment->trace() << "\t" << (flag_exploration?"Learning":"Testing") << endl;
			}

			//! XCS ends the current problem
			xcs->end_problem();
			
			//! the environment ends the current problem
			environment->end_problem();
			
			//! problem statistics are saved
			/*! by default the statistics file contain (for each line)
			 *  - experiment number
			 *  - problem number
			 *  - number of problem steps
			 *  - total reward gained during the problem
			 *  - population size
			 *  - "Learning/Testing" whether the problem has been solved in learning or testing mode
			 */

			if (statistics_rolling_window==0)
			{
				//! saves the statistics for each problem
				update_statistics(STATISTICS, (flag_exploration?"Learning":"Testing"), is_single_step,
											current_experiment, current_problem,
											problem_steps, reward_sum, xcs->no_macroclassifiers(),
											system_error);
			} else {
				if (!flag_exploration)
				{
					update_rolling_statistics(STATISTICS, rolling_statistics,
                                       is_single_step, current_experiment,
                                       current_problem, first_learning_problem,
                                       problem_steps, reward_sum,
                                       xcs->no_macroclassifiers(), system_error);
				}
			}

			//! it switches from exploration to exploitation and viceversa
			flag_exploration = !flag_exploration;

			//--------------------------------------------------------------------------------
			//! save intermediate populations
			//--------------------------------------------------------------------------------

			unsigned long no_problems_so_far = current_problem-first_learning_problem;

			if ((no_problems_so_far>0) && save_population_interval!=0)
			{
				// cout << "SAVE POPULATION INTERVAL " << save_population_interval << endl;
				if (no_problems_so_far%save_population_interval==0)
				{
					save_population((current_experiment), current_problem);
				}
			}

		} //!< end learning/testing problems

        is_condensation_active = false;

		//! save the stats collected by XCS (# GA activations, ...)
		char fn_xcs_statistics[max_filename_size];		//! filename statistics file
		snprintf(fn_xcs_statistics, max_filename_size, "execution_statistics.%s-%04ld", extension.c_str(), current_experiment);
		xcs_statistics stats = xcs->statistics();
		stats.save(fn_xcs_statistics);


		//--------------------------------------------------------------------------------
		// Save Timing Report
		//--------------------------------------------------------------------------------
		experiment_timer.compute_statistics();
		save_time_report(experiment_timer,current_experiment);

		//--------------------------------------------------------------------------------
		// Evaluate the Environment
		// XCS solves a problem starting from each possible initial configuration
		//--------------------------------------------------------------------------------
		
		unsigned long current_test_problem = (statistics_rolling_window==0)?current_problem:current_problem/2;

		if (environment->allow_test() && flag_test_environment)
		{
			map<string,vector<double>> rolling_solution_statistics {
				{"Steps", vector<double>()},
				{"Reward", vector<double>()},
				{"Macroclassifier", vector<double>()},
				{"System Error", vector<double>()},
			};

			environment->reset_input();
			flag_exploration = false;

			do 
			{
				//! we don't call environment->begin_problem() since it is managed in reset_input()

				//! init XCS for the problem starts (does not do anything)
	            xcs->begin_problem();

				solve_episode(/* is_exploration_problem = */ false, /* is_condensation_active = */ false, problem_steps, reward_sum, system_error);

				if (flag_trace) 
				{
					TRACE << current_experiment << "\t" << current_test_problem << '\t' << "<<" << xcs->trace() << "\t" << environment->trace() << "\t" << "Solution" << endl;
				}

				//! XCS ends the current problem
				xcs->end_problem();
		
				//! the environment ends the current problem
				environment->end_problem();

				if (statistics_rolling_window==0)
				{
					update_statistics(STATISTICS, "Solution", is_single_step,
												current_experiment, current_test_problem,
												problem_steps, reward_sum, xcs->no_macroclassifiers(),
												system_error);
				} else {

					rolling_solution_statistics["Steps"].push_back(problem_steps);
					rolling_solution_statistics["Reward"].push_back(reward_sum);
					rolling_solution_statistics["Macroclassifier"].push_back(xcs->no_macroclassifiers());

					if (is_single_step)
					{
						rolling_solution_statistics["System Error"].push_back(system_error);
					}
				}

				current_test_problem++;

			} while (environment->next_input());

			if (statistics_rolling_window!=0)
			{
				save_solution_statistics(STATISTICS, rolling_solution_statistics, current_experiment, current_test_problem, is_single_step);		
			}
		}
		
		//--------------------------------------------------------------------------------
		// Save the Action-Value Function
		//--------------------------------------------------------------------------------
		save_avf(current_experiment);

		//! XCS ends the experiment
		xcs->end_experiment();
		
		//! at the end of the experiment the file for statistics is closed and gzipped
		STATISTICS.close();
		snprintf(system_command, max_system_command_size, "gzip -f %s", fn_statistics);
		system(system_command);
	
		if (flag_trace)
		{
			TRACE.close();
			snprintf(system_command, max_system_command_size, "gzip -f %s", fn_trace);
			system(system_command);
		}

		if (save_population_interval!=0)
		{
			save_population((current_experiment), current_problem);
		}

		if (flag_save_final_population) 
		{
			save_population(current_experiment);
		}
	}


}

// solves one episode starting from an initial state
void 
experiment_mgr2::solve_episode(bool is_exploration_episode, bool is_condensation_active, unsigned long &no_steps, double &total_reward, double &system_error)
{

	//! system_error should be >0; -1.0 -> not updated during episode
	system_error = -1.0;

    // //! init XCS for the problem starts (does not do anything)
    // xcs->begin_problem();

    // //! init the environment for the current problem
    // environment->begin_problem();

	no_steps = 0;
	total_reward = 0;
    do {
        //! XCS executes one step
        xcs->step(is_exploration_episode, is_condensation_active);
        no_steps++;

        //! sum up the reward received
        total_reward += environment->reward();

        // if teletransportation is active then restart after a certain amount of steps
        // - teletransportation only works during learning
        // - it is not activated at the end of the experiment

        // bool flag_can_teletransport = flag_exploration || XCS->update_during_test_problems();
        bool flag_can_teletransport = is_exploration_episode && teletransportation_interval > 0;

        if (flag_can_teletransport && !environment->stop()) {
            if ((no_steps > 0) && (no_steps % teletransportation_interval == 0)) {
                xcs->begin_problem();
                environment->begin_problem();
            }
        }
    } while ((no_steps < no_max_steps) && (!environment->stop()));

	if (environment->is_single_step())
		system_error = xcs->get_system_error();	
}


void 
experiment_mgr2::save_time_report(const xcslib::timing &experiment_timer, unsigned long experiment)
{
	if (!flag_save_time_report)
		return;

    //! init the file for statistics
    ofstream REPORT;

	char filename_report[max_filename_size];
    snprintf(filename_report, max_filename_size, "timing-report.%s-%04ld", extension.c_str(), experiment);

    REPORT.open(filename_report, ios::out | ios::app);
    if (!REPORT.good())
    {
		xcs_utility::error(class_name(), "save_time_report", "Report file '"+string(filename_report)+"' not open", 1);
    }

	experiment_timer.save_report(REPORT,extension+"\t"+xcs_utility::datetime());

    REPORT.close();
};

void	
experiment_mgr2::save_population(unsigned long current_experiment, unsigned long problem_no) const
{
	ofstream	POPULATION;
	char		filename[max_filename_size];
        char system_command[max_system_command_size];

        clog << "\t" << current_experiment+1 << "/" << first_experiment+no_experiments << "\t";

	if (problem_no==0)
	{	
		snprintf(filename, max_filename_size, "population.%s-%04d", extension.c_str(), (int) current_experiment);
		clog << "saving the final population ...         ";
	} else {
		snprintf(filename, max_filename_size, "population.%s-%04d-%015ld", extension.c_str(), (int) current_experiment, problem_no);
		clog << "saving the population at " << std::setfill ('0') << std::setw (15) << problem_no;
	}

	POPULATION.open(filename);

	if (!POPULATION.good())
	{
		xcs_utility::error(class_name(),"save_agent", "Population file " + string(filename) + " not open", 1);
	}

	xcs->save_population(POPULATION);

	POPULATION.close();
	
	snprintf(system_command, max_system_command_size, "gzip -f %s", filename);
	system(system_command);
			
	clog << "\t\tok" << endl;
}

void experiment_mgr2::set_parameters(xcslib::configuration_manager &xcs_config)
{
    try
    {
        first_experiment = xcs_config.Value(tag_name(), "first experiment");
	} catch (...)
    {
        xcs_utility::error(class_name(), "constructor", "attribute \'first experiment\' not found in <" + tag_name() + ">", 1);
    }

	try
    {
        no_experiments = xcs_config.Value(tag_name(), "number of experiments");
	} catch (...)
    {
        xcs_utility::error(class_name(), "constructor", "attribute \'number of experiments\' not found in <" + tag_name() + ">", 1);
    }

	// try
    // {
	first_learning_problem = xcs_config.Value(tag_name(), "first problem", (unsigned long) 0);
	// } catch (...)
    // {
    //     xcs_utility::error(class_name(), "constructor", "attribute \'first problem\' not found in <" + tag_name() + ">", 1);
    // }

	try
    {
        no_learning_problems = xcs_config.Value(tag_name(), "number of learning problems");
	} catch (...)
    {
        xcs_utility::error(class_name(), "constructor", "attribute \'number of learning problems\' not found in <" + tag_name() + ">", 1);
    }

	no_condensation_problems = xcs_config.Value(tag_name(), "number of condensation problems", (unsigned long) 0);
	// try
    // {
    //     no_condensation_problems = xcs_config.Value(tag_name(), "number of condensation problems", 0);
	// } catch (...)
    // {
    //     xcs_utility::error(class_name(), "constructor", "attribute \'number of condensation problems\' not found in <" + tag_name() + ">", 1);
    // }

	try
    {
		no_test_problems = xcs_config.Value(tag_name(), "number of test problems", (unsigned long)0);

	} catch (...)
    {
        xcs_utility::error(class_name(), "constructor", "attribute \'number of test problems\' not found in <" + tag_name() + ">", 1);
    }

	//! optional parameters

	//! maximum number of steps
	no_max_steps = xcs_config.Value(tag_name(), "maximum number of steps", (unsigned long)1500);

	//! save populations
	string str_save_population = (string)xcs_config.Value(tag_name(), "save final population", "on");
	xcs_utility::set_flag(string(str_save_population), flag_save_final_population);	
	save_population_interval = xcs_config.Value(tag_name(), "save population every", (unsigned long)0);

	//! save experiment state
	string str_save_experiment_state = (string)xcs_config.Value(tag_name(), "save experiment final state", "off");
	xcs_utility::set_flag(string(str_save_experiment_state), flag_save_experiment_final_state);
	save_experiment_interval = xcs_config.Value(tag_name(), "save experiment state every", (unsigned long)0);

	//! save problem trace (e.g., sequence of positions in multistep problems)
    string str_save_trace = (string)xcs_config.Value(tag_name(), "save problem execution trace", "off");
    xcs_utility::set_flag(string(str_save_trace), flag_trace);

	//! teletransportation interval; if zero, teletransportation is disabled
    teletransportation_interval = (unsigned long)xcs_config.Value(tag_name(), "teletransportation interval", (unsigned long)0);
	if (teletransportation_interval != 0 && teletransportation_interval < 3)
	{
		xcs_utility::error(class_name(), "constructor", "Teletransportation interval must be at least 3", 1);
	}
    
	// string str_test_environment = (string)xcs_config.Value(tag_name(), "test environment", "off");
    // xcs_utility::set_flag(string(str_test_environment), flag_test_environment);
	xcs_utility::set_flag(xcs_config.Value(tag_name(), "evaluate solution", "off"), flag_test_environment);

    //! saves execution time
	// string str_trace_time = (string)xcs_config.Value(tag_name(), "trace time", "on");
	// xcs_utility::set_flag(string(str_trace_time), flag_save_time_report);
	xcs_utility::set_flag(xcs_config.Value(tag_name(), "save execution time report", "on"), flag_save_time_report);	

    //! saves action value function
	xcs_utility::set_flag(xcs_config.Value(tag_name(), "save action-value function", "off"), flag_save_avf);

	//! optional parameters
	statistics_rolling_window = xcs_config.Value(tag_name(), "statistics rolling window size", (unsigned long)0);																						 

	if (statistics_rolling_window!=0)
	{
		if (statistics_rolling_window%100!=0)
		{
	        xcs_utility::warning(class_name(), "constructor", "attribute \'statistics rolling window\' should be a multiple of 100.");
		}
	}
}

void experiment_mgr2::print_parameters(ostream& OUTPUT)
{
	OUTPUT << "<" << tag_name() << ">" << endl;
	OUTPUT << "\t" << "first experiment = "<< first_experiment << endl;
	OUTPUT << "\t" << "number of experiments = " << no_experiments << endl;
	OUTPUT << "\t" << "first problem = " << first_learning_problem << endl;
	OUTPUT << "\t" << "number of learning problems = " << no_learning_problems << endl;
	OUTPUT << "\t" << "number of condensation problems = " << no_condensation_problems << endl;
	OUTPUT << "\t" << "number of test problems = " << no_test_problems << endl;
	OUTPUT << "\t" << "save final population = " << (flag_save_final_population?"on":"off") << endl;
	OUTPUT << "\t" << "save population every = " << save_population_interval << endl;
	OUTPUT << "\t" << "save experiment final state = " << (flag_save_experiment_final_state?"on":"off") << endl;
	OUTPUT << "\t" << "save experiment state every = " << save_experiment_interval << endl;
	OUTPUT << "\t" << "save problem execution trace = " << (flag_trace?"on":"off") << endl;
	OUTPUT << "\t" << "evaluate solution = " << (flag_test_environment?"on":"off") << endl;
	OUTPUT << "\t" << "maximum number of steps = " << no_max_steps << endl;
	OUTPUT << "\t" << "teletransportation interval = " << teletransportation_interval << endl;
	OUTPUT << "\t" << "save execution time report = " << (flag_save_time_report?"on":"off") << endl;
	OUTPUT << "\t" << "save action-value function = " << (flag_save_avf?"on":"off") << endl;
	OUTPUT << "</" << tag_name() << ">" << endl;
}

void 
experiment_mgr2::save_avf(const unsigned long expNo, const unsigned long problem_no) const
{
	if (!flag_save_avf || !environment->allow_test())
		return;

	char filename[max_filename_size];
	char system_command[max_system_command_size];

	ofstream AVF;
	t_action action;

	if (problem_no==0)
	{
		clog << "\t" << current_experiment+1 << "/" << first_experiment+no_experiments << "\t";
		clog << "saving the final action-value function ...";
	}

	unsigned long no_actions = action.actions();

	if (problem_no==0)
		snprintf(filename, max_filename_size, "avf.%s-%04d", extension.c_str(), (int) current_experiment);
	else 
		snprintf(filename, max_filename_size, "avf.%s-%04d-%015ld", extension.c_str(), (int) current_experiment, problem_no);

	AVF.open(filename);

	if (!AVF.good())
	{
		xcs_utility::error(class_name(),"save_agent", "Population file " + string(filename) + " not open", 1);
	}

	//! column names
	AVF << "State";
	for (int i=0; i<no_actions;i++)
	{
		AVF << "|"<< t_action(i);
	}
	AVF << endl;

	environment->reset_problem();

	do {
		vector<double> prediction = xcs->predict(environment->state());
		AVF << environment->state();
		for (int i=0; i<no_actions;i++)
		{
			AVF << "|"<< prediction[i];
		}
		AVF << endl;
	} while (environment->next_problem());
	AVF.close();
	
	snprintf(system_command, max_system_command_size, "gzip -f %s", filename);
	system(system_command);

	if (problem_no==0)
		clog << "\tok" << endl;
}

void experiment_mgr2::update_statistics(ostream& STATISTICS, const string& label, bool is_single_step, unsigned long current_experiment, unsigned long current_problem, unsigned long no_problem_steps, double reward_sum, unsigned long no_macroclassifiers, double system_error)
const
{
	//! problem statistics are saved
	/*! by default the statistics file contain (for each line)
		*  - experiment number
		*  - problem number
		*  - number of problem steps
		*  - total reward gained during the problem
		*  - population size
		*  - "Learning/Testing/Solution" whether the problem has been solved in learning, testing, or it is evaluating the final solution
		*/

	STATISTICS << current_experiment << '\t' << current_problem << '\t';
	STATISTICS << no_problem_steps << '\t';
	STATISTICS << reward_sum << '\t';
	STATISTICS << no_macroclassifiers << '\t';
	if (is_single_step)
	{
		STATISTICS << system_error << "\t";
	}
	STATISTICS << label << endl;
}

void experiment_mgr2::update_rolling_statistics(ostream &STATISTICS, map<string, vector<double>> &rolling_statistics,
                                               bool is_single_step,
                                               unsigned long current_experiment, unsigned long current_problem,
                                               unsigned long first_learning_problem, unsigned long no_problem_steps,
                                               double reward_sum, unsigned long no_macroclassifiers,
                                               double system_error)
{
	unsigned long current_no_test_problems = (current_problem - first_learning_problem)/2;
    unsigned long position_in_rolling_window = current_no_test_problems % statistics_rolling_window;

    rolling_statistics["Steps"][position_in_rolling_window] = no_problem_steps;
    rolling_statistics["Reward"][position_in_rolling_window] = reward_sum;
    rolling_statistics["Macroclassifier"][position_in_rolling_window] = no_macroclassifiers;
    if (is_single_step) 
	{
        rolling_statistics["System Error"][position_in_rolling_window] = system_error;
    }

	if (position_in_rolling_window + 1 == statistics_rolling_window) 
	{
        STATISTICS << current_experiment << '\t' << (current_problem + 1)/2;

		double average_steps =
            std::reduce(rolling_statistics["Steps"].begin(), rolling_statistics["Steps"].end(), 0.0) /
            double(statistics_rolling_window);

        double average_reward_sum =
            std::reduce(rolling_statistics["Reward"].begin(), rolling_statistics["Reward"].end(), 0.0) /
            double(statistics_rolling_window);

        double average_no_macroclassifiers = std::reduce(rolling_statistics["Macroclassifier"].begin(),
                                                         rolling_statistics["Macroclassifier"].end(), 0.0) /
                                             double(statistics_rolling_window);
		
        STATISTICS << "\t" << std::fixed << setprecision(5) << average_steps;
        STATISTICS << "\t" << std::fixed << setprecision(5) << average_reward_sum;
        STATISTICS << "\t" << std::fixed << setprecision(2) << average_no_macroclassifiers;
        if (is_single_step) {
            double average_system_error =
                std::reduce(rolling_statistics["System Error"].begin(), rolling_statistics["System Error"].end(), 0.0) /
                double(statistics_rolling_window);

            STATISTICS << "\t" << std::fixed << setprecision(5) << average_system_error;
        }
        STATISTICS << "\t" << "Testing" << endl;
    }
}

void 
experiment_mgr2::save_solution_statistics(ofstream &STATISTICS, map<string,vector<double>> rolling_solution_statistics, unsigned long current_experiment, unsigned long reported_problem, bool is_single_step)
const
{
	STATISTICS << current_experiment << '\t' << reported_problem;

	double average_steps =
		std::reduce(rolling_solution_statistics["Steps"].begin(), rolling_solution_statistics["Steps"].end(), 0.0) /
		double(rolling_solution_statistics["Steps"].size());

	double average_reward_sum =
		std::reduce(rolling_solution_statistics["Reward"].begin(), rolling_solution_statistics["Reward"].end(), 0.0) /
		double(rolling_solution_statistics["Reward"].size());

	double average_no_macroclassifiers = std::reduce(rolling_solution_statistics["Macroclassifier"].begin(),
													rolling_solution_statistics["Macroclassifier"].end(), 0.0) /
										double(rolling_solution_statistics["Macroclassifier"].size());
	
	STATISTICS << "\t" << std::fixed << setprecision(5) << average_steps;
	STATISTICS << "\t" << std::fixed << setprecision(5) << average_reward_sum;
	STATISTICS << "\t" << std::fixed << setprecision(2) << average_no_macroclassifiers;

	if (is_single_step) 
	{
		double average_system_error =
			std::reduce(rolling_solution_statistics["System Error"].begin(), rolling_solution_statistics["System Error"].end(), 0.0) /
			double(rolling_solution_statistics["System Error"].size());

		STATISTICS << "\t" << std::fixed << setprecision(5) << average_system_error;
	}
	STATISTICS << "\t" << "Solution" << endl;
}