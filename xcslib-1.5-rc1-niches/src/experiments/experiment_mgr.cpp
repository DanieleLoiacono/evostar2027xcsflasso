#include "xcs_utility.h"
#include "timer.h"
#include "experiment_mgr.h"
#include "xcs_definitions.h"


/*!
 * \file experiment_mgr.cpp
 *
 * \brief implements the methods for the experiment manager 
 *
 */

const std::vector<std::string> experiment_mgr::configuration_parameters = {"first experiment","number of experiments","first problem","number of learning problems","number of condensation problems","number of test problems","maximum number of steps","save final population","save population every","save experiment final state","save experiment state every","save problem execution trace","teletransportation interval","test environment","save execution time report", "save action-value function"};

experiment_mgr::experiment_mgr(xcslib::configuration_manager &xcs_config, t_classifier_system *xcs, t_environment *environment, bool verbose)
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

    // PLL Not sure what these were used for
	// current_experiment = -1;	//! to check whether the method reset_experiments is called
	// current_problem = -1;		//! to check whether the method reset_problems is called
}


void
experiment_mgr::perform_experiments()
{
    char system_command[max_system_command_size]; //! string used to perform system calls.

    char fn_statistics[max_filename_size]; //! filename statistics file
    char fn_trace[max_filename_size];      //! filename trace file
    ofstream STATISTICS; //! files that contains the whole experiment statitics. One line for each problem performed.
    ofstream TRACE;      //! files that contains the trace information about the experiment

    double reward_sum = 0;  //! sum of rewards gained while solving the problem
    long problem_steps = 0; //! number of steps needed to solve the problem

    xcslib::timer timer_overall;    //! measure the CPU time for the all the experiments
    xcslib::timer timer_experiment; //! measure the CPU time for one experiment
    xcslib::timer timer_problem;    //! measure the CPU time for one problem

    vector<double> experiment_time; //! time elapsed for each experiment
    vector<double> problem_time;    //! time elapsed for problems

    double average_problem_time;     //! average time for problems
    double average_learning_time;    //! average time for learning
    double average_testing_time;     //! average time for testing
    bool flag_compact_stats_printed; //! true if the compact line was just printed

    experiment_time.clear();
    problem_time.clear();
    timer_overall.start();

    //! performs all the experiments, one by one.
    for (current_experiment = first_experiment; current_experiment < (first_experiment + no_experiments);
         current_experiment++) {

        //! true if condensation is active
        bool flag_condensation = false;

        //! init XCS for the current experiment
        xcs->begin_experiment();

        //! init the environment for the current experiment
        environment->begin_experiment();

        //! the first problem is always solved in exploration
        bool flag_exploration = true;

        //! init the file for statistics
        snprintf(fn_statistics, max_filename_size, "statistics.%s-%04ld", extension.c_str(), current_experiment);

        /*!
         * if first_learning_problem is greater than 0 indicates that the experiment must be restored from file;
         * otherwise the experiment starts from scratch.
         */
        if (first_learning_problem > 0) {
            //! restore from the current experiment
            cout << "\nRestarting Experiment " << current_experiment;
            cout << "... " << endl;

            //! experiments statistics will be appened to existing files
            snprintf(system_command, max_filename_size, "gunzip %s.gz", fn_statistics);
            system(system_command);
            STATISTICS.open(fn_statistics, ios::out | ios::app);

            //! restores the state of the current experiment
            flag_exploration = restore_state(current_experiment);
        }
        else {
            //! init the statistics file for a new experiment
            STATISTICS.open(fn_statistics);
        };

        if (!STATISTICS.good()) {
            xcs_utility::error(class_name(), "perform_experiments",
                               "Statistics file '" + string(fn_statistics) + "' not open", 1);
        }

        snprintf(fn_trace, max_filename_size, "trace.%s-%04ld", extension.c_str(), current_experiment);

        if (flag_trace) {
            /*!
             * if first_learning_problem is greater than 0 indicates that the experiment must be restored from file;
             * otherwise the experiment starts from scratch.
             */
            if (first_learning_problem > 0) {
                snprintf(system_command, max_filename_size, "gunzip %s.gz", fn_trace);
                system(system_command);
                TRACE.open(fn_trace, ios::out | ios::app);
            }
            else { //! create a new trace file
                TRACE.open(fn_trace);
            }
            if (!TRACE.good()) {
                char errMsg[max_filename_size] = "";
                snprintf(errMsg, max_filename_size, "Trace file '%s' not open", fn_trace);
                xcs_utility::error(class_name(), "StartSession", string(errMsg), 1);
            }
        }

        //! start timer for the experiment
        timer_experiment.start();
        average_problem_time = 0;

        current_no_test_problems = 0;

        //! performs the learning problems one by one
        for (current_problem = first_learning_problem;
             current_problem <
             first_learning_problem + 2 * (no_learning_problems + no_condensation_problems) + no_test_problems;
             current_problem++) {
            STATISTICS << current_experiment << '\t' << current_problem << '\t';

            //! if needed save information in the trace file
            // if (flag_trace)
            // {
            // 	TRACE << current_experiment << "\t" << current_problem << '\t';
            // }

            //! start timer for problem
            timer_problem.start();

            //! init XCS for the current problem
            xcs->begin_problem();

            //! determine whether condensation should be activated
            flag_condensation = ((no_condensation_problems > 0) &&
                                 (current_problem >= first_learning_problem + 2 * no_learning_problems));

            //! if learning has ended, the problems are performed in testing mode
            if (current_problem >= (first_learning_problem + 2 * (no_learning_problems + no_condensation_problems))) {
                flag_exploration = false;
            }

            //! init the environment for the current problem
            environment->begin_problem();

            reward_sum = 0;
            problem_steps = 0;
            do {
                //! XCS executes one step
                xcs->step(flag_exploration, flag_condensation);
                problem_steps++;

                //! sum up the reward received
                reward_sum = reward_sum + environment->reward();

                // if teletransportation is active then restart after a certain amount of steps
                // - teletransportation only works during learning
                // - it is not activated at the end of the experiment

                // bool flag_can_teletransport = flag_exploration || XCS->update_during_test_problems();
                bool flag_can_teletransport = flag_exploration && teletransportation_interval > 0;

                if (flag_can_teletransport && !environment->stop()) {
                    if ((problem_steps > 0) && (problem_steps % teletransportation_interval == 0)) {
                        xcs->begin_problem();
                        environment->begin_problem();
                    }
                }
            } while ((problem_steps < no_max_steps) && (!environment->stop()));

            //! stops the timer for the problem
            timer_problem.stop();
            average_problem_time += timer_problem.elapsed();

            //! problem trace information is saved
            /*! by default the statistics file contain (for each line)
             *  - experiment number
             *  - problem number
             *  - trace information from XCS (usually null)
             *  - trace information from the environment
             *  - "Learning/Testing" whether the problem has been solved in learning or testing mode
             */

            if (flag_trace) {
                TRACE << current_experiment << "\t" << current_problem;
                TRACE << "\t" << xcs->trace();
                TRACE << "\t" << environment->trace();
                TRACE << (flag_exploration ? "Learning" : "Testing") << endl;
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

            STATISTICS << problem_steps << '\t';
            STATISTICS << reward_sum << '\t';
            STATISTICS << xcs->no_macroclassifiers() << '\t';

            //! if the environment is single step, it saves the system_error
            if (environment->is_single_step()) {
                STATISTICS << xcs->get_system_error() << "\t";
            }
            STATISTICS << (flag_exploration ? "Learning" : "Testing") << endl;
            // }

            //! it switches from exploration to exploitation and viceversa
            flag_exploration = !flag_exploration;

            //! save intermediate experiment states
            unsigned long no_problems_so_far = current_problem - first_learning_problem;

            if ((no_problems_so_far > 0) && save_experiment_interval != 0) {
                if (no_problems_so_far % save_experiment_interval == 0) {
                    save_state((current_experiment), flag_exploration, current_problem);
                }
            }

            //! save intermediate populations
            if ((no_problems_so_far > 0) && save_population_interval != 0) {
                // cout << "SAVE POPULATION INTERVAL " << save_population_interval << endl;
                if (no_problems_so_far % save_population_interval == 0) {
                    save_population((current_experiment), current_problem);
                }
            }

            if (!flag_exploration) {
                current_no_test_problems++;
                flag_compact_stats_printed = false;
            }

        } //!< end learning/testing problems

        //! stops the experimnt timer
        timer_experiment.stop();

        //! memorize the time used in this experiment
        experiment_time.push_back(timer_experiment.elapsed());
        problem_time.push_back(average_problem_time /
                               (no_learning_problems + no_condensation_problems + no_test_problems));

        /*!
         *
         * Test the environment
         *
         * for each possible initial configuration of the environment
         * XCS is applied until the problem's end
         *
         */

        if (environment->allow_test() && flag_test_environment) {
            environment->reset_problem();

            do {
                xcs->begin_problem();

                //! when testing the environment
                flag_exploration = false;
                flag_condensation = false;
                ///==============================================================================
                /*!
                 * write the number of experiment and problem to the files
                 */

                //! save information in the statistics file
                STATISTICS << current_experiment << '\t' << current_problem << '\t';

                // //! if needed save information in the trace file
                // if (flag_trace)
                // {
                // 	TRACE << current_experiment << "\t" << current_problem << '\t';
                // }

                reward_sum = 0;
                problem_steps = 0;
                do {
                    //! if more than 5000 steps have been performed, the systems is forced to explore
                    if (problem_steps > no_max_steps) {
                        cerr << ">> Maximum number of steps reached during testing the environment. Exploration "
                                "activated."
                             << endl;
                        flag_exploration = true;
                    }

                    //! XCS executes one step
                    xcs->step(flag_exploration, flag_condensation);
                    problem_steps++;

                    //! sum up the reward received
                    reward_sum = reward_sum + environment->reward();

                } while (!environment->stop());

                //! problem trace information is saved
                /*! by default the statistics file contain (for each line)
                 *  - experiment number
                 *  - problem number
                 *  - trace information from XCS (usually null)
                 *  - trace information from the environment
                 *  - "Learning/Testing" whether the problem has been solved in learning or testing mode
                 */

                if (flag_trace) {
                    TRACE << current_experiment << "\t" << current_problem;
                    TRACE << "\t" << xcs->trace();
                    TRACE << "\t" << environment->trace();
                    TRACE << (flag_exploration ? "Learning" : "Testing") << endl;
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

                STATISTICS << problem_steps << '\t';
                STATISTICS << reward_sum << '\t';
                STATISTICS << xcs->no_macroclassifiers() << '\t';
                //! if the environment is single step, it saves the system_error
                if (environment->is_single_step()) {
                    STATISTICS << xcs->get_system_error() << "\t";
                }
                STATISTICS << (flag_exploration ? "Learning" : "Solution") << endl;

                ///==============================================================================
                current_problem++;

            } while (environment->next_problem());
        }

        if (environment->allow_test() && flag_save_avf) {
            save_avf(current_experiment);
        }

        //! stop the timer for the whole session
        timer_overall.stop();

        //! XCS ends the experiment
        xcs->end_experiment();

        //! at the end of the experiment the file for statistics is closed and gzipped
        STATISTICS.close();
        snprintf(system_command, max_filename_size, "gzip -f %s", fn_statistics);
        system(system_command);

        if (flag_trace) {
            TRACE.close();
            snprintf(system_command, max_filename_size, "gzip -f %s", fn_trace);
            system(system_command);
        }

        //! save requested information about the experiment.
        if (flag_save_experiment_final_state) {
            save_state(current_experiment, flag_exploration);
        }

        if (save_population_interval != 0) {
            save_population((current_experiment), current_problem);
        }

        if (flag_save_final_population) {
            save_population(current_experiment);
        }

	}

	if (flag_save_time_report)
	{
        save_time_report(timer_overall, experiment_time, problem_time);
    }
}

void experiment_mgr::save_time_report(xcslib::timer &timer_overall, std::vector<double> &experiment_time, std::vector<double> &problem_time)
{
    //! init the file for statistics
    ofstream REPORT;
    char filename_report[max_filename_size];

    snprintf(filename_report, max_filename_size, "report.%s-%04ld", extension.c_str(), current_experiment);

    REPORT.open(filename_report, ios::out | ios::app);
    if (!REPORT.good())
    {
		xcs_utility::error(class_name(), "save_time_report", "Report file '"+string(filename_report)+"' not open", 1);
    }

    REPORT << "TOTAL ELAPSED TIME\t\t" << setprecision(4) << timer_overall.elapsed() << endl;
    for (unsigned long exp = first_experiment; exp < (first_experiment + no_experiments); exp++)
    {
        REPORT << "Experiment\t" << setw(5) << exp << "\t";
        REPORT << "Total\t" << experiment_time[exp - first_experiment] << "\t";
        REPORT << "AveragePerProblem\t" << problem_time[exp - first_experiment] << "\t";
        REPORT << endl;
    }

    REPORT << "----------------------------------------------------------------------------------------------------" << endl;
    REPORT << endl << endl;
    REPORT.close();
};

void	
experiment_mgr::print_save_options(ostream& output) 
const
{
	output << "\nEXPERIMENT MANAGER SAVE OPTIONS\n" << endl;
			
	output << "\tsave final population:\t\t" << (flag_save_final_population ? "yes" : "no") << endl;

	if (save_population_interval!=0)
		output << "\tsave population every:\t\t" << save_population_interval << endl;

	output << "\tsave experiment final state:\t" << (flag_save_experiment_final_state ? "yes" : "no") << endl;

	if (save_experiment_interval!=0)
		output << "\tsave experiment state every:\t" << save_experiment_interval << endl;											
};

void	
experiment_mgr::save_population(const unsigned long current_experiment, const unsigned long problem_no) const
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
	
	snprintf(system_command, max_filename_size, "gzip -f %s", filename);
	system(system_command);
			
	clog << "\t\tok" << endl;
}

void	
experiment_mgr::save_state(const unsigned long expNo, const bool flag_exploration, unsigned long problem_no) const
{
	ofstream	OUTPUT;
	char		filename[max_filename_size];
        char system_command[max_system_command_size]; // string for system calls

        clog << "\t" << current_experiment+1 << "/" << first_experiment+no_experiments << "\t";
	clog << "saving the experiment final state ...";

	if (problem_no==0)
		snprintf(filename, max_filename_size, "experiment.%s-%04d", extension.c_str(), (int) expNo);
	else 
		snprintf(filename, max_filename_size, "experiment.%s-%04d-%015ld", extension.c_str(), (int) expNo, problem_no);

	OUTPUT.open(filename);
	if (!OUTPUT.good())
	{
		char errMsg[max_filename_size] = "";
		snprintf(errMsg, max_filename_size, "Experiment's state file '%s' not created", filename);
		xcs_utility::error(class_name(),"save_state",string(errMsg),1);
	}
	else
	{
		OUTPUT << flag_exploration;
		OUTPUT << endl;
		xcs_random::save_state(OUTPUT);
		OUTPUT << endl;
		environment->save_state(OUTPUT);
		OUTPUT << endl;
		xcs->save_state(OUTPUT);
		OUTPUT << endl;

		OUTPUT.close();
		
		snprintf(system_command, max_filename_size, "gzip -f %s", filename);
		system(system_command);

		clog << "\t\tok" << endl;
	}
}

bool
experiment_mgr::restore_state(const unsigned long expNo)
{
	cout << "Restoring system state ... ";
	ifstream	infile;
	char	fileName[max_filename_size] = "";
        char system_command[max_system_command_size]; // string for system calls
        bool	flag_exploration;
	
	snprintf(fileName, max_filename_size, "experiment.%s-%d", extension.c_str(), (int) expNo);
        snprintf(system_command, max_filename_size, "gunzip %s", fileName);
        system(system_command);
        infile.open(fileName);

	if (!infile.good())
	{
		char errMsg[max_filename_size] = "";
		snprintf(errMsg, max_filename_size,
				"Experiment's state file '%s' not open",
				fileName);
		xcs_utility::error(class_name(),"restore_state",string(errMsg),1);
	}

	infile >> flag_exploration;
	xcs_random::restore_state(infile);
	environment->restore_state(infile);	
	xcs->restore_state(infile);
	infile.close();

	snprintf(system_command, max_filename_size, "gzip -f %s", fileName);
	system(system_command);
	cout << "Ok\n" << endl;
	return flag_exploration;
}


void experiment_mgr::set_parameters(xcslib::configuration_manager &xcs_config)
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

	try
    {
        no_condensation_problems = xcs_config.Value(tag_name(), "number of condensation problems");
	} catch (...)
    {
        xcs_utility::error(class_name(), "constructor", "attribute \'number of condensation problems\' not found in <" + tag_name() + ">", 1);
    }

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
	xcs_utility::set_flag(xcs_config.Value(tag_name(), "test environment", "off"), flag_test_environment);

    //! saves execution time
	// string str_trace_time = (string)xcs_config.Value(tag_name(), "trace time", "on");
	// xcs_utility::set_flag(string(str_trace_time), flag_save_time_report);
	xcs_utility::set_flag(xcs_config.Value(tag_name(), "save execution time report", "on"), flag_save_time_report);	

    //! saves action value function
	xcs_utility::set_flag(xcs_config.Value(tag_name(), "save action-value function", "off"), flag_save_avf);
}

void experiment_mgr::print_parameters(ostream& OUTPUT)
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
	OUTPUT << "\t" << "test environment = " << (flag_test_environment?"on":"off") << endl;
	OUTPUT << "\t" << "maximum number of steps = " << no_max_steps << endl;
	OUTPUT << "\t" << "teletransportation interval = " << teletransportation_interval << endl;
	OUTPUT << "\t" << "save execution time report = " << (flag_save_time_report?"on":"off") << endl;
	OUTPUT << "\t" << "save action-value function = " << (flag_save_avf?"on":"off") << endl;
	OUTPUT << "</" << tag_name() << ">" << endl;
}

void 
experiment_mgr::save_avf(const unsigned long expNo, const unsigned long problem_no) const
{
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
	
	snprintf(system_command, max_filename_size, "gzip -f %s", filename);
	system(system_command);

	if (problem_no==0)
		clog << "\tok" << endl;
}

