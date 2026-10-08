/*! 
 *  \file	xcsf_classifier_system.cpp
 *  \brief	Implementation of the XCSF classifier system as described in Butz & Wilson paper.
 *
 */

#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "xcsf_classifier_system.h"
#include "high_resolution_timer.h"
#include "timer.h"

using namespace std;

const std::vector<std::string> xcsf_classifier_system::configuration_parameters = {"population size", "epsilon zero", "theta GA", "initial population", "crossover probability", "mutation probability", "learning rate", "discount factor", "discovery component", "vi", "alpha", "prediction init", "error init", "fitness init", "set size init", "exploration strategy", "theta delete", "theta GA sub", "theta AS sub", "GA subsumption", "GA subsumption on [A]", "AS subsumption", "update during test", "update error first", "gradient descent", "niche queue max size", "offspring selection for GA", "offspring selection for condensation"};
// "deletion strategy", 

xcsf_classifier_system::xcsf_classifier_system(xcslib::configuration_manager& xcs_config, t_environment *environment)
{
	this->environment = environment;

	//! look for the init section in the configuration file
	if (!xcs_config.exist(tag_name()))
	{
		xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
	}

	// check if the configuration file contains only allowed keywords
	xcs_config.check_parameters(tag_name(), configuration_parameters);

	//! set the parameters from the configuration file
    set_parameters(xcs_config);

    //! reserve memory for [P], [M], [A], [A]-1
	match_set.reserve(max_population_size);
	action_set.reserve(max_population_size);
	previous_action_set.reserve(max_population_size);
	select.reserve(max_population_size);

	//! create the prediction array
	create_prediction_array();
	
	//! check subsumption settings
	t_condition	cond;

	if (flag_as_subsumption && !cond.allow_as_subsumption())
	{
		xcs_utility::error( class_name(), "constructor", "AS subsumption requested but condition class does not allow", 1);
	}

	if (flag_ga_subsumption && !cond.allow_ga_subsumption())
	{
		xcs_utility::error( class_name(), "constructor", "GA subsumption requested but condition class does not allow", 1);
	}	
}

void xcsf_classifier_system::set_parameters(xcslib::configuration_manager &xcs_config)
{
	try 
	{
		max_population_size = xcs_config.Value(tag_name(), "population size");
	} catch (...) {
        xcs_utility::error(class_name(), "constructor", "attribute \'population size\' not found in <" + tag_name() + ">", 1);
	}

	try 
	{
        epsilon_zero = xcs_config.Value(tag_name(), "epsilon zero");
	} catch (...) {
        xcs_utility::error(class_name(), "constructor", "attribute \'epsilon zero\' not found in <" + tag_name() + ">", 1);
	}

	// str_discovery_component = (string)xcs_config.Value(tag_name(), "discovery component", "on");
	xcs_utility::set_flag(xcs_config.Value(tag_name(), "discovery component", "on"), flag_discovery_component);

	// if the discovery component it is used, it is mandatory to specify its parameters
	if (flag_discovery_component)
	{
		try 
		{
			theta_ga = xcs_config.Value(tag_name(), "theta GA");

		} catch (...) {
			xcs_utility::error(class_name(), "constructor", "attribute \'theta GA\' not found in <" + tag_name() + ">", 1);
		}

		try 
		{
			prob_crossover = xcs_config.Value(tag_name(), "crossover probability");
		} catch (...) {
			xcs_utility::error(class_name(), "constructor", "attribute \'crossover probability\' not found in <" + tag_name() + ">", 1);
		}

		try 
		{
			prob_mutation = xcs_config.Value(tag_name(), "mutation probability");
		} catch (...) {
			xcs_utility::error(class_name(), "constructor", "attribute \'mutation probability\' not found in <" + tag_name() + ">", 1);
		}
	} else {
		// discovery component is off so the parameters are optional and if not set they have default values
		theta_ga = xcs_config.Value(tag_name(), "theta GA", 25.0);
		prob_crossover = xcs_config.Value(tag_name(), "crossover probability", 0.8);
		prob_mutation = xcs_config.Value(tag_name(), "mutation probability", 0.04);
	}

	// if not specified use roulette wheel
	string str_ga_offspring_selection = (string) xcs_config.Value(tag_name(), "offspring selection for GA", "roulette-wheel");
	set_offspring_selection_strategy(str_ga_offspring_selection, ga_offspring_selection_strategy, ga_tournament_size);
	
	// if not specified, condensation use the same selection strategy used for the discovery component
	string str_condensation_offspring_selection = (string) xcs_config.Value(tag_name(), "offspring selection for condensation", str_ga_offspring_selection);
	set_offspring_selection_strategy(str_condensation_offspring_selection, condensation_offspring_selection_strategy, condensation_tournament_size);


#ifdef __NICHE_TRACKING__
	try {
		max_niche_queue_size = (unsigned long) xcs_config.Value(tag_name(), "niche queue max size");
		clog << "niche queue max size = " << max_niche_queue_size << endl;
	} catch (const char *attribute) {
		if (max_population_size==0)
		{
			xcs_utility::error(class_name(), "constructor", "missing population size cannot set \'niche queue max size\' in <" + tag_name() + ">", 1);
		}
		max_niche_queue_size = (unsigned long) (.10*max_population_size);
		clog << "attribute 'niche queue max size' not found in <" + tag_name() + "> set to 10\% of |[P]| = "<< max_niche_queue_size << endl;
	}
#endif

	//! typical values that never change in all configurations
	learning_rate = xcs_config.Value(tag_name(), "learning rate", 0.2);
	discount_factor = xcs_config.Value(tag_name(), "discount factor",0.7);

	vi = xcs_config.Value(tag_name(), "vi", 5.0);
	alpha = xcs_config.Value(tag_name(), "alpha", 0.1);

	init_prediction = xcs_config.Value(tag_name(), "prediction init", 10.0);
	init_error = xcs_config.Value(tag_name(), "error init", 0.0);
	init_fitness = xcs_config.Value(tag_name(), "fitness init", 0.01);
	init_set_size = xcs_config.Value(tag_name(), "set size init", 1.0);

	// str_pop_init = (string)xcs_config.Value(tag_name(), "initial population", "empty");
    set_init_strategy(xcs_config.Value(tag_name(), "initial population", "empty"));

	string str_exploration = (string)xcs_config.Value(tag_name(), "exploration strategy", "random");
    set_exploration_strategy(str_exploration.c_str());

	theta_del = xcs_config.Value(tag_name(), "theta delete", 20.0);
	theta_sub = xcs_config.Value(tag_name(), "theta GA sub", 20.0);
	theta_as_sub = xcs_config.Value(tag_name(), "theta AS sub", 100.0);

    xcs_utility::set_flag(xcs_config.Value(tag_name(), "GA subsumption", "off"), flag_ga_subsumption);

	t_condition	condition;

	if (!condition.allow_ga_subsumption() && flag_ga_subsumption)
	{
		xcs_utility::error(class_name(),"xcsf_classifier_system",
			"conditions <"+condition.class_name()+"> does not allow GA subsumption which is set to on in the parameters.",1);
	}

	//! if the GA Subsumption is activated also the GAA Subsubmption (introduced in the XCS algorithmic description is activated)
	string str_gaa_sub = (string)xcs_config.Value(tag_name(), "GA subsumption on [A]", flag_ga_subsumption?"on":"off");
    xcs_utility::set_flag(string(str_gaa_sub), flag_gaa_subsumption);

	// string str_as_sub = (string)xcs_config.Value(tag_name(), "AS subsumption", "off");
    xcs_utility::set_flag(xcs_config.Value(tag_name(), "AS subsumption", "off"), flag_as_subsumption);

    string str_update_test = (string)xcs_config.Value(tag_name(), "update during test", "on");
	xcs_utility::set_flag(string(str_update_test), flag_update_test);

    // str_error_first = (string)xcs_config.Value(tag_name(), "update error first", "on");
    xcs_utility::set_flag(xcs_config.Value(tag_name(), "update error first", "on"), flag_error_update_first);

    // string str_use_mam = (string)xcs_config.Value(tag_name(), "use MAM", "on");
    xcs_utility::set_flag(xcs_config.Value(tag_name(), "use MAM", "on"), flag_use_mam);

    // string str_use_gd = (string)xcs_config.Value(tag_name(), "gradient descent", "off");
	xcs_utility::set_flag(xcs_config.Value(tag_name(), "gradient descent", "off"), flag_use_gradient_descent);

	//! constant parameters 
	delta_del = 0.1;
}

void xcsf_classifier_system::print_parameters(ostream& OUTPUT) const
{
	OUTPUT << "<" << tag_name() << ">" << endl;
	OUTPUT << "\t" << "population size = " << max_population_size << endl;
	OUTPUT << "\t" << "epsilon zero = " << epsilon_zero << endl;
	OUTPUT << "\t" << "theta GA = " << theta_ga << endl;
	OUTPUT << "\t" << "crossover probability = " << prob_crossover << endl;
	OUTPUT << "\t" << "mutation probability = " << prob_mutation << endl;
	OUTPUT << "\t" << "learning rate = " << learning_rate << endl;
	OUTPUT << "\t" << "discount factor = " << discount_factor << endl;
	OUTPUT << "\t" << "discovery component = " << (flag_discovery_component?"on":"off") << endl;
	OUTPUT << "\t" << "vi = " << vi << endl;
	OUTPUT << "\t" << "alpha = " << alpha << endl;
	OUTPUT << "\t" << "prediction init = " << init_prediction << endl;
	OUTPUT << "\t" << "error init = " << init_error << endl;
	OUTPUT << "\t" << "fitness init = " << init_fitness << endl;
	OUTPUT << "\t" << "set size init = " << init_set_size << endl;

	string str_exploration_strategy;
	stringstream str_epsilon;

	switch (action_selection_strategy)
	{
		case action_selection::epsilon_greedy:
			str_epsilon << std::fixed << std::setprecision(2) << prob_random_action;
			str_exploration_strategy = "epsilon-greedy " + str_epsilon.str();
			break;
		case action_selection::random:
			str_exploration_strategy = "random";
			break;
		case action_selection::greedy:
			str_exploration_strategy = "deterministic";
			break;
	}

	OUTPUT << "\t" << "exploration strategy = " << str_exploration_strategy << endl;

	string str_deletion_strategy; 

	OUTPUT << "\t" << "deletion strategy = " << str_deletion_strategy << endl;

	OUTPUT << "\t" << "theta delete = " << theta_del << endl;
	OUTPUT << "\t" << "theta GA sub = " << theta_sub << endl;
	OUTPUT << "\t" << "theta AS sub = " << theta_as_sub << endl;

	OUTPUT << "\t" << "GA subsumption = " << (flag_ga_subsumption?"on":"off") << endl;
	OUTPUT << "\t" << "GAA subsumption = " << (flag_gaa_subsumption?"on":"off") << endl;
	OUTPUT << "\t" << "AS subsumption = " << (flag_as_subsumption?"on":"off") << endl;

	OUTPUT << "\t" << "update during test = " << (flag_update_test?"on":"off") << endl;
	OUTPUT << "\t" << "update error first = " << (flag_error_update_first?"on":"off") << endl;

	OUTPUT << "\t" << "use MAM = " << (flag_use_mam?"on":"off") << endl;
	OUTPUT << "\t" << "update error first = " << (flag_error_update_first?"on":"off") << endl;

	// OUTPUT << "\t" << "tournament selection = " << (flag_ga_tournament_selection?"on":"off") << endl;
	// OUTPUT << "\t" << "tournament size = " << tournament_size << endl;

	OUTPUT << "\t" << "gradient descent = " << (flag_use_gradient_descent?"on":"off") << endl;

	// MUST ADD THE SELECTION METHOD
}

void	
xcsf_classifier_system::set_exploration_strategy(const string& exploration_strategy)
{
	if (exploration_strategy=="random") {
		action_selection_strategy = action_selection::random;
	} else if (exploration_strategy=="deterministic") {
		action_selection_strategy = action_selection::greedy;
	} else if (xcs_utility::trim(exploration_strategy.substr(0,exploration_strategy.find(" ")))=="epsilon-greedy") {
		action_selection_strategy = action_selection::epsilon_greedy;
		string str_epsilon = xcs_utility::trim(exploration_strategy.substr(exploration_strategy.find(" ")+1));
		prob_random_action = std::stof(str_epsilon);

		if ((prob_random_action<=0.) || (prob_random_action>1.))
		{	
			xcs_utility::error(class_name(),"set_exploration_strategy",
				"epsilon-greedy value out of range (0.0,1.0]",1);
		}
	} else {
		xcs_utility::error(class_name(),"set_exploration_strategy",
			"Unrecognized action selection policy \'"+exploration_strategy+"\'",1);
	}
}

void
xcsf_classifier_system::init_classifier_set()
{
	// t_classifier classifier;
	switch (population_init)
	{
		//! [P] = {}
		case population_initialization::empty:
			clear_population();
			break;

		//! fill [P] with random classifiers
		case population_initialization::random:
			init_population_random();
			break;

		//! fill [P] with classifiers save in a file
		case population_initialization::load:
			init_population_load(population_init_file);
			break;

		//! fill [P] with classifiers save in a file
		case population_initialization::solution:
			init_population_solution(population_init_file);
			break;

		default:
			xcs_utility::error(class_name(),"init_classifier_set", "init strategy unknown" , 1);
	}
};
				

bool compare_classifiers(t_classifier *clp1, t_classifier *clp2) {return *clp1==*clp2;};
bool compare_cl(t_classifier *clp1, t_classifier *clp2) {return *clp1<*clp2;};

xcsf_classifier_system::t_set_const_iterator
xcsf_classifier_system::find_classifier(const t_classifier_set& population, const t_classifier& classifier) 
const
{
	t_set_const_iterator pp = population.begin();

	while (pp!=population.end())
	{
		if ((**pp)==classifier)
		{
		 	return pp;
		}
		pp++;
	}
	return pp;
}

void
xcsf_classifier_system::insert_classifier(const t_classifier &classifier)
{
	t_set_const_iterator pp = find_classifier(population, classifier);
	
	if (pp!=population.end())
	{
		(*pp)->numerosity++;

		#ifdef __NICHE_TRACKING__
		(*pp)->creation_time_stamp = total_time;
		(*pp)->as_time_stamp = 0;
		(*pp)->init_time_stamps();
		#endif

	} else {
		t_classifier_ptr classifier_ptr = new t_classifier(classifier);
		classifier_ptr->time_stamp = total_steps;
		classifier_ptr->experience = 0;
		classifier_ptr->generate_id();

		// cout << "#1 " << classifier << endl;
		// cout << "#2 " << *classifier_ptr << endl;

		#ifdef __NICHE_TRACKING__
		classifier_ptr->creation_time_stamp = total_time;
		classifier_ptr->as_time_stamp = 0;
		classifier_ptr->init_time_stamps();
		#endif

		population.push_back(classifier_ptr);
		macro_size++;	
		
		stats.no_insert_classifiers++;
	}

	population_size++;

}

//! build [M]
unsigned long	
xcsf_classifier_system::match(const t_state& detectors)
{
	t_set_iterator			pp;		/// iterator for visiting [P]
	unsigned long			match_set_size = 0;		/// number of micro classifiers in [M]

	match_set.clear();				/// [M] = {}

#ifdef __FAST_BINARY_MATCHING__
        istringstream   INPUTS(detectors.string_value()); 
        bitset<__BIT_CONDITION_SIZE__>    input;

        INPUTS >> input;

        for(pp=population.begin();pp!=population.end();pp++)
        {  

                if ((**pp).condition.match(input))
                {     
                        match_set.push_back(*pp); 
                        sz += (**pp).numerosity;
                }     
        }  
#else 	
	for(pp=population.begin();pp!=population.end();pp++)
	{
		if ((**pp).match(detectors))
		{
			match_set.push_back(*pp);
			match_set_size += (**pp).numerosity;
       	}
   	}
#endif

	return match_set_size;
}

void xcsf_classifier_system::build_prediction_array(const vector<double> &current_input)
{
	t_set_iterator					mp;
	vector<t_system_prediction>::iterator		pr;	
	t_system_prediction				prediction;

	//! clear P(.)
	init_prediction_array();

	//! scan [M] and build the prediction array
	for(mp=match_set.begin(); mp!=match_set.end(); mp++ )
	{
            double computed_prediction = (**mp).get_prediction(current_input);

            //!	look whether the action was already found in the prediction array
            pr = find(prediction_array.begin(), prediction_array.end(), ((**mp).action));

            if (pr == prediction_array.end()) {
                xcs_utility::error(class_name(), "build_prediction_array", "action not found in prediction array", 1);
                exit(-1);
                /*!	the action was not previously found
                 *	thus prediction array is initialized with the
                 *	classifier values
                 */
                prediction.payoff = computed_prediction * (**mp).fitness;
                prediction.sum = (**mp).fitness;
                prediction.n = 1;
                //!	add the element to the prediction array
                prediction_array.push_back(prediction);
		} else {  
			/*!	the action was already in the array
			 *	thus the corresponding value is updated 
			 *	with the classifier values
			 */
                        pr->payoff += computed_prediction * (**mp).fitness;
                        pr->sum += (**mp).fitness;
			pr->n++;
		}
	}

	available_actions.clear();
	for(pr=prediction_array.begin(); pr!=prediction_array.end(); pr++ )
	{
		if (pr->n!=0)
		{
			available_actions.push_back((pr - prediction_array.begin()));
			pr->payoff /= pr->sum;
		}
	}
};

t_action	
xcsf_classifier_system::select_action(const action_selection policy)
const
{
	assert(available_actions.size()>0);

	switch(policy)
	{
		//! selects a random action
		case action_selection::random:
			return select_random_action();
			break;

		//! select the action with the highest payoff
		case action_selection::greedy:
			return select_best_action();
			break;
			
		//! random action with probability prob_random_action, best action otherwise
		case action_selection::epsilon_greedy:
			if (xcs_random::random()<prob_random_action)
				return select_random_action();
			else
				return select_best_action();
			
			break;

		default: 
			xcs_utility::error(class_name(),"select_action", "action selection strategy not allowed", 1);
	};
}


t_action 
xcsf_classifier_system::select_best_action() 
const
{
	assert(available_actions.size()>0);

	// select best without shuffling
	long no_actions = available_actions.size();

	long random_action_index = xcs_random::dice(no_actions);

	long best_action_index = random_action_index;

	for(int i=1; i<no_actions; i++)
	{
		long next_action_index = (random_action_index+i)%no_actions;

		if (prediction_array[best_action_index].payoff<=prediction_array[next_action_index].payoff)
		{
			best_action_index = next_action_index;
		}
	}

	return prediction_array[best_action_index].action;
}

t_action 
xcsf_classifier_system::select_random_action() 
const 
{
	return prediction_array[available_actions[xcs_random::dice(available_actions.size())]].action;
}

void xcsf_classifier_system::update_set(const double P, t_classifier_set &action_set,
                                        const vector<double> &current_input)
{
	t_set_iterator	clp;
	double		set_size = 0;
	double		fitness_sum = 0;	//! sum of classifier fitness in [A]

	//! update the experience of classifiers in [A]
	//! estimate the action set size
	for(clp=action_set.begin(); clp != action_set.end(); clp++)
	{
		(**clp).experience++;
		set_size += (**clp).numerosity;
		fitness_sum += (**clp).fitness;	//! sums up classifier fitness for gradient descent
	}

	for(clp=action_set.begin(); clp!= action_set.end(); clp++)
	{
            double computed_prediction = (**clp).get_prediction(current_input);

            //! prediction error is updated first if required (i.e., flag_error_update is true)
            if (flag_error_update_first) {
                //! update the classifier prediction error
                if (!flag_use_mam || ((**clp).experience > (1 / learning_rate))) {
                    (**clp).error += learning_rate * (fabs(P - computed_prediction) - (**clp).error);
                }
                else {
                    (**clp).error += (fabs(P - computed_prediction) - (**clp).error) / (**clp).experience;
                }
            }

            (**clp).update_prediction(current_input, P);
            computed_prediction = (**clp).get_prediction(current_input);

            if (!flag_error_update_first) {
                //! update the classifier prediction error
                if (!flag_use_mam || ((**clp).experience > (1 / learning_rate))) {
                    (**clp).error += learning_rate * (fabs(P - computed_prediction) - (**clp).error);
                }
                else {
                    (**clp).error += (fabs(P - computed_prediction) - (**clp).error) / (**clp).experience;
                }
            }

                //! update the classifier action set size estimate
		if (!flag_use_mam || ((**clp).experience>(1/learning_rate)))
		{
			(**clp).actionset_size += learning_rate*(set_size - (**clp).actionset_size);
		} else {
			(**clp).actionset_size += (set_size - (**clp).actionset_size)/(**clp).experience;
		}
	}

	//! update fitness
	update_fitness(action_set);
}

void
xcsf_classifier_system::update_fitness(t_classifier_set &action_set)
{
	t_set_iterator 			as;
	double				ra;
	vector<double>			raw_accuracy;
	vector<double>::iterator	rp;
	double				accuracy_sum = 0;

	raw_accuracy.clear();

	for(as = action_set.begin(); as !=action_set.end(); as++)
	{
		if ((**as).error<epsilon_zero)
			ra = (**as).numerosity;
		else 
			ra = alpha*(pow(((**as).error/epsilon_zero),-vi)) * (**as).numerosity;

		raw_accuracy.push_back(ra);
		accuracy_sum += ra;
	}

	for(as = action_set.begin(), rp=raw_accuracy.begin(); as!=action_set.end(); as++,rp++)
	{
		(**as).fitness += learning_rate*((*rp)/accuracy_sum - (**as).fitness);
	}

}

bool
xcsf_classifier_system::subsume(const t_classifier &first, const t_classifier &second)
{
	bool	result;
	
	result = (classifier_could_subsume(first, epsilon_zero, theta_sub)) && (first.subsume(second));

	if (result)
		stats.no_subsumption++;
	
	return result;
}

bool
xcsf_classifier_system::need_ga(t_classifier_set &action_set, const bool flag_explore)
{
	double		average_set_stamp = 0;
	unsigned long	size = 0;

	if (!flag_explore) return false;

	t_set_iterator 	as;	

	for(as=action_set.begin(); as!=action_set.end(); as++)
	{
		average_set_stamp += (**as).time_stamp * (**as).numerosity;
		size += (**as).numerosity;
	}

	average_set_stamp = average_set_stamp / size;

	if (total_steps<average_set_stamp)
	{
		cout << "TOTSTEPS = " << total_steps << endl;
		cout << "AVGTS = " << average_set_stamp << endl;
	}

	if (total_steps<average_set_stamp)
		cerr << "NEEDGA " << total_steps << " " << average_set_stamp << endl; 
	assert(total_steps>=average_set_stamp);
	return ((total_steps - average_set_stamp)>=theta_ga);
}

void xcsf_classifier_system::update_timestamp(t_classifier_set &action_set)
{
	//! set the time stamp of classifiers in [A]
	for(t_set_iterator as=action_set.begin(); as!=action_set.end(); as++)
	{
		(**as).time_stamp = total_steps;
	}
}


void
xcsf_classifier_system::select_offsprings(t_classifier_set &action_set, t_classifier_ptr &parent1, t_classifier_ptr &parent2)
{
	switch (ga_offspring_selection_strategy)
	{
		case offspring_selection::roulette_wheel:
			roulette_wheel_selection(action_set, parent1, parent2);
			break;

		case offspring_selection::tournament_selection:
			tournament_selection(action_set, parent1, ga_tournament_size);
			tournament_selection(action_set, parent2, ga_tournament_size);
			break;
	}
}


void
xcsf_classifier_system::genetic_algorithm(t_classifier_set &action_set, const t_state& detectors)
{
	t_classifier_ptr	parent1;
	t_classifier_ptr	parent2;

	update_timestamp(action_set);

	select_offsprings(action_set, parent1, parent2);

	
	t_classifier	offspring1 = *parent1;
	t_classifier	offspring2 = *parent2;

	offspring1.numerosity = offspring2.numerosity = 1;
	offspring1.experience = offspring2.experience = 1;

	if (xcs_random::random()<prob_crossover)
	{
		offspring1.recombine(offspring2);
		offspring1.prediction = offspring2.prediction = (parent1->prediction+parent2->prediction)/2;
		offspring1.error = offspring2.error = (parent1->error+parent2->error)/2;
		offspring1.fitness = offspring2.fitness = (parent1->fitness+parent2->fitness)/2;
		offspring1.time_stamp = offspring2.time_stamp = total_steps;
		offspring1.actionset_size = offspring2.actionset_size = (parent1->actionset_size + parent2->actionset_size)/2;

#ifdef __NICHE_TRACKING__
		offspring1.creation_time_stamp = offspring2.creation_time_stamp = total_time;
		offspring1.as_time_stamp = offspring2.as_time_stamp = 0;
		offspring1.init_time_stamps();
		offspring2.init_time_stamps();				
#endif
		stats.no_crossover++;
	}

	offspring1.mutate(prob_mutation,detectors);
	offspring2.mutate(prob_mutation,detectors);

	offspring1.fitness = 0.1 * offspring1.fitness;
	offspring2.fitness = 0.1 * offspring2.fitness;;

	if (flag_ga_subsumption)
	{
		//! if the offspring was subsumed and the parent numerosity increase, 
		//! delete the offspring, otherwise insert the offspring in the population				
		if (!apply_subsumption(offspring1,parent1,parent2,action_set))
		{
			insert_classifier(offspring1);
		}

		if (!apply_subsumption(offspring2,parent1,parent2,action_set))
		{
			insert_classifier(offspring2);
		}

	} else {
		//! no subsumption, just insert the offspring in the population
		insert_classifier(offspring1);
		insert_classifier(offspring2);

	}
	
	//! whatever happened, two classifiers have been added to the population
	//! so at most two classifiers must be deleted from it.
	delete_classifier();
	delete_classifier();
}

//! check if the offspring is subsumed by one of the parents or one of the classifiers in the action set
bool	
xcsf_classifier_system::apply_subsumption(const t_classifier &offspring, t_classifier_ptr parent1, t_classifier_ptr parent2, t_classifier_set &action_set)
{
	//! parent1 subsumes the offspring
	if (subsume(*parent1, offspring))
	{	
		parent1->numerosity++;
		population_size++;

		// offspring has been subsumed so it must be deleted
		return true;
	}

	//! parent2 subsumes the offspring
	if (subsume(*parent2, offspring))
	{	
		parent2->numerosity++;
		population_size++;

		// offspring has been subsumed so it must be deleted
		return true;
	} 

	//! offspring is not subsumed by the parents. if the subsumption against the action set is activated do it
	if (flag_gaa_subsumption)
	{
		return subsumed_by_the_classifier_set(offspring, action_set);
	}

	//! all subsumptions failed
	return false;
}


//! if Martin's GA subsumption is used, offspring classifier is compared to the classifiers in [A]
bool	
xcsf_classifier_system::subsumed_by_the_classifier_set(const t_classifier& classifier, t_classifier_set &action_set)
{
	t_set_iterator 	subsuming_classifier_iterator = find_subsuming_classifier(classifier, action_set);
	if (subsuming_classifier_iterator!=action_set.end())
	{				
		(**subsuming_classifier_iterator).numerosity++;
		population_size++;
		return true;
	} else {
		return false;
	}
}

void
xcsf_classifier_system::condensation(t_classifier_set &action_set)
{
	t_classifier_ptr 	parent1;
	t_classifier_ptr	parent2;

	update_timestamp(action_set);

	select_offsprings(action_set, parent1, parent2);
	
	//! does not mutate nor recombine but it checks whether they are subsumed in the action set if the [A] subsubmption is active
	if (flag_gaa_subsumption)
	{
		//! if parent1 has not been subsumed by the action set, then increase its numerosity
		if (!subsumed_by_the_classifier_set(*parent1, action_set))
		{
			stats.no_subsumptions_condensation++;
			parent1->numerosity++;
			population_size++;

		}

        //! if parent2 has not been subsumed by the action set, then increase its numerosity
		if (!subsumed_by_the_classifier_set(*parent2, action_set))
		{
			stats.no_subsumptions_condensation++;
			parent2->numerosity++;
			population_size++;
		}
	} else {
		stats.no_subsumptions_condensation++;
		parent1->numerosity++;
		population_size++;

		stats.no_subsumptions_condensation++;
		parent2->numerosity++;
		population_size++;
	}

	//! whatever happened two classifiers have been added so delete two if needed
	delete_classifier();
	delete_classifier();
}

void	
xcsf_classifier_system::step(const bool exploration_mode, const bool condensationMode)
{
	t_action		action;				//! selected action
	unsigned long	match_set_size;		//! number of microclassifiers in [M]
	double			P;					//! value for prediction update, computed as r + gamma * max P(.) 
	double			max_prediction;

	//! save the state before performing the action
	t_state saved_state;
	vector<double> saved_state_numeric;

	//! reads the current input
	current_input = environment->state(); 
	vector<double> current_input_numeric = current_input.numeric_representation();

	//! update the number of learning steps performed so far
	if (exploration_mode)
	{
		total_steps++;
		total_learning_steps++;
	} 

	total_time++;

	/*! 
	 * check if [M] needs covering,
	 * if it does, it apply the selected covering strategy, i.e., standard as defined in Wilson 1995,
	 * or action_based as defined in Butz and Wilson 2001
	 */

	do {
		match_set_size = match(current_input);
		build_prediction_array(current_input_numeric);
	} while (covering(current_input));

	//! build the prediction array P(.)

	build_prediction_array(current_input_numeric);

#ifdef __DEBUG__
	cout << "BUILT THE PREDICTION ARRAY" << endl;
	print_prediction_array(cout);
	cout << endl;
#endif
	
	//! select the action to be performed
	if (exploration_mode)
		action = select_action(action_selection_strategy);
	else 
		action = select_action(action_selection::greedy);

	//! build [A]
	build_action_set(action);

	//! store the current input before performing the selected action	
	saved_state = environment->state();
	saved_state_numeric = previous_input.numeric_representation();

	//! perform the selected action in the environment
	environment->perform(action);

	//! if the environment is single step, the system error is collected
	if (environment->is_single_step())
	{
		double payoff = prediction_array[action.value()].payoff;
		system_error = fabs(payoff-environment->reward());
	}

	total_reward = total_reward + environment->reward();

	//! reinforcement component
	
	//! if [A]-1 is not empty it computes P
	if ((exploration_mode || flag_update_test) && previous_action_set.size())
	{
		vector<t_system_prediction>::iterator	pr = prediction_array.begin();
		max_prediction = pr->payoff;

		for(pr = prediction_array.begin(); pr!=prediction_array.end(); pr++)
		{
			if (max_prediction<pr->payoff)
			{
				max_prediction = pr->payoff;
			}
		}

		double expected_payoff = previous_reward + discount_factor * max_prediction;
		update_set(expected_payoff, previous_action_set, previous_input_numeric);
	}

	if (environment->stop())
	{
		if (exploration_mode || flag_update_test)
		{
			double expected_payoff = environment->reward();

			//! if the episode is over, the last action set is updated with the final reward using the last input 
			update_set(expected_payoff, action_set, current_input_numeric);
		}
	}

	//! apply the genetic algorithm to [A] if needed
	if (flag_discovery_component && need_ga(action_set, exploration_mode))
	{
		if (condensationMode)
		{
			condensation(action_set);
			stats.no_condensation++;
		} else {
			genetic_algorithm(action_set, current_input);
			stats.no_ga++;
		}		
	}
	
	//!	[A]-1 <= [A]
	//!	r-1 <= r
	previous_action_set = action_set;
	previous_reward = environment->reward();
	previous_input = current_input;
	previous_input_numeric = current_input_numeric;	
	action_set.clear();
}

void	
xcsf_classifier_system::save_population(ostream& output) 
{
	t_set_iterator		pp;
	t_classifier_set 	save;

	for(pp=population.begin(); pp!=population.end(); pp++)
	{
		output << (**pp) << endl;
	}
}

void	
xcsf_classifier_system::save_state(ostream& output) 
{
	output << stats << endl;
	output << total_steps << endl;
	t_classifier::save_state(output);
	output << macro_size << endl;

	save_population(output);
}

void
xcsf_classifier_system::restore_state(istream& input)
{
	unsigned long size;
	input >> stats;
	input >> total_steps;
	t_classifier::restore_state(input);
	input >> size;
	
	population.clear();
	
    t_classifier in_classifier;
	population_size = 0;
	macro_size = 0;

	for(unsigned long cl=0; cl<size; cl++)
	{
		if (!input.eof() && (input >> in_classifier))
		{
			t_classifier	*classifier = new t_classifier(in_classifier);
			population.push_back(classifier);
			population_size += classifier->numerosity;
			macro_size++;
		}
	};
	assert(macro_size==size);
}

//! defines what has to be done when a new experiment begins
void 
xcsf_classifier_system::begin_experiment()
{
	//! reset the overall time step
	total_time = 0;

	//! reset the total number of learning steps
	total_learning_steps = 0;

	//! reset the total number of steps
	total_steps = 0;

	//! reset the counter of classifiers ids
	t_classifier::reset_id();

	//! init the experiment statistics
	stats.reset();
	
	//! [P] contains 0 macro/micro classifiers
	population_size = 0;
	macro_size = 0;

	//! init [P]
	init_classifier_set();
}

//! operations needed at the end of the experiment
void 
xcsf_classifier_system::end_experiment() 
{
	cout << endl;
	cout << "EXPERIMENT STATISTICS (BEGIN)" << endl;
	cout << "----------------------------------------------------------------------" << endl;
	stats.pretty_print(cout);
	cout << "----------------------------------------------------------------------" << endl;
	cout << "EXPERIMENT STATISTICS (END)" << endl;
	cout << endl;
}


//! defines what has to be done when a new problem begins. 
void
xcsf_classifier_system::begin_problem()
{
	//! clean [M]
	match_set.clear();

	//! clear [A]-1
	previous_action_set.clear();
	
	//! clear [A]
	action_set.clear();

	//! set the steps within the problem to 0
	problem_steps = 0;

	//! clear the total reward gained
	total_reward = 0;
}

//! defines what must be done when the current problem ends
void 
xcsf_classifier_system::end_problem() 
{
	match_set.clear();
	action_set.clear();
	previous_action_set.clear();
}

bool
xcsf_classifier_system::covering(const t_state& detectors)
{
	//! true => some covering classifiers have been created
	bool covered_some_actions = false;

	for(vector<t_system_prediction>::const_iterator pr=prediction_array.begin(); pr!=prediction_array.end(); pr++ )
	{
		if (pr->n==0)
		{
			t_classifier classifier;

			clog << "COVERING " << detectors << endl;
			classifier.cover(detectors);
			classifier.action = pr->action;

			init_classifier(classifier);
			
			insert_classifier(classifier);

			delete_classifier();

			covered_some_actions = true;
		}
	}
	return covered_some_actions;
};

void
xcsf_classifier_system::init_classifier(t_classifier& classifier)
{
	classifier.prediction = init_prediction;
	classifier.error = init_error;
	classifier.fitness = init_fitness;

	classifier.actionset_size = init_set_size;
	classifier.experience = 0;
	classifier.time_stamp = total_steps;

	classifier.numerosity = 1;

#ifdef __NICHE_TRACKING__
	classifier.creation_time_stamp = total_time;
	classifier.as_time_stamp = 0;
	classifier.init_time_stamps();
#endif

}

//! build [A] from [M] and an action "act"
/*!
 * \param action selected action
 */
void	
xcsf_classifier_system::build_action_set(const t_action& action)
{
	//! iterator in [M]
	t_set_iterator 	mp;

	//! clear [A]
	action_set.clear();

	//! fill [A] with the classifiers in [M] that have action "act"
	for( mp=match_set.begin(); mp!=match_set.end(); mp++ )
	{
		//if ((**mp).action==action)
		if ((*mp)->action==action)
		//if (true)
		{

#ifdef __NICHE_TRACKING__
			(*mp)->as_time_stamp = total_time;
			if (max_niche_queue_size>0)
			{
				(*mp)->update_time_stamps(total_time, max_niche_queue_size);
			}
#endif
			action_set.push_back(*mp);
		}
	}

	// std::shuffle(action_set.begin(), action_set.end(), xcs_random::rng());
}

//! clear [P]
void
xcsf_classifier_system::clear_population()
{
	//! iterator in [P]
	t_set_iterator	pp;

	//! delete all the classifiers in [P]
	for(pp=population.begin(); pp!=population.end(); pp++)
	{
		delete *pp; 
	}

	//! delete all the pointers in [P]
	population.clear();

	//! number of macro classifiers is set to 0
	macro_size = 0;

	//! number of micro classifiers is set to 0
	population_size = 0;
}

//! print a set of classifiers
/*!
 * \param set set of classifiers
 * \bug cause a memory leak!
 */
void 
xcsf_classifier_system::print_set(t_classifier_set &set, ostream& output) 
{
	t_set_const_iterator	pp;

	output << "================================================================================" << endl;
	for(pp=set.begin(); pp!=set.end(); pp++)
	{
		output << (**pp);
		output << endl;
	}
	output << "================================================================================" << endl;
}

//@{
//! set the strategy to init [P] at the beginning of the experiment
/*! 
 * \param strategy can be either \emph empty or \emph random
 * two strategies are allowed \emph empty set [P] to the empty set
 * \emph random fills [P] with random classifiers
 */
void
xcsf_classifier_system::set_init_strategy(string strategy)
{

#ifdef __DEBUG__
	cout << "STRATEGY " << strategy << endl;
	cout << "PREFIX " << strategy.substr(0,5)<< endl;
	cout << "SUFFIX " << strategy.substr(5)<< endl;
#endif

	if ( (strategy!="random") && (strategy!="empty") && (strategy.substr(0,5)!="load:") && (strategy.substr(0,9)!="solution:"))
	{	
		//!	unrecognized deletion strategy
		string	msg = "Unrecognized population init policy";
		xcs_utility::error(class_name(),"set_init_strategy", msg, 1);
	}

	if (strategy=="empty")
		population_init = population_initialization::empty;
	else if (strategy=="random")
		population_init = population_initialization::random;
	else if (strategy.substr(0,5)=="load:")
	{
		population_init = population_initialization::load;
		population_init_file = strategy.substr(5);
		cout << "LOAD FROM <" << population_init_file << ">" << endl;
	} else if (strategy.substr(0,9)=="solution:")
	{
		population_init = population_initialization::solution;
		population_init_file = strategy.substr(9);
		// cout << "INIT POPULATION WITH SOLUTION FROM <" << population_init_file << ">" << endl;
	} else {
		//!	unrecognized deletion strategy
		string	msg = "Unrecognized population init policy";
		xcs_utility::error(class_name(),"set_init_strategy", msg, 1);
	}
}

void
xcsf_classifier_system::set_offspring_selection_strategy(const string& str_offspring_selection_strategy, offspring_selection &selection_strategy, double &tournament_size)
{
	cout << "<<" << str_offspring_selection_strategy << ">>" << endl;

	if (xcs_utility::trim(str_offspring_selection_strategy) == "roulette-wheel")
	{
		selection_strategy = offspring_selection::roulette_wheel;
	} else if (xcs_utility::trim(str_offspring_selection_strategy.substr(0,str_offspring_selection_strategy.find(" ")))=="tournament-selection")
	{
		string str_tournament_size = xcs_utility::trim(str_offspring_selection_strategy.substr(str_offspring_selection_strategy.find(" ")+1));
		
		selection_strategy = offspring_selection::tournament_selection;
		tournament_size = std::stof(str_tournament_size);

		if ((tournament_size<=0.) || (tournament_size>1.))
		{	
			xcs_utility::error(class_name(),"set_offspring_selection_strategy", "tournament size out of range (0.0,1.0]",1);
		}
	} else {
		xcs_utility::error(class_name(),"set_offspring_selection_strategy",
			"Unrecognized offspring selection policy \'"+str_offspring_selection_strategy+"\'",1);
	}
}

void
xcsf_classifier_system::roulette_wheel_selection(t_classifier_set &action_set, t_classifier_ptr &clp1, t_classifier_ptr &clp2)
{
	vector<double>	roulette_wheel;
	double fitness_sum = 0;

	for(t_set_const_iterator as=action_set.begin(); as!=action_set.end(); as++)
	{
		fitness_sum += (**as).fitness;
		roulette_wheel.push_back(fitness_sum);
	}

	//! 
	double random1 = (xcs_random::random())*fitness_sum;
	double random2 = (xcs_random::random())*fitness_sum;

	if (random1>random2)
		swap(random1,random2);

	t_set_iterator test1 = select_proportional(action_set, roulette_wheel, random1);
	t_set_iterator test2 = select_proportional(action_set, roulette_wheel, random2);

	// clp1 = (*test1);
	// clp2 = (*test2);

	// this is just to check whether procedure with binary search is correct
	unsigned long selected_parent_i = 0;
	for(selected_parent_i = 0; (selected_parent_i<roulette_wheel.size()) && (random1>=roulette_wheel[selected_parent_i]); selected_parent_i++);
	clp1 = action_set[selected_parent_i];
	assert(selected_parent_i<roulette_wheel.size());

	for(selected_parent_i = 0; (selected_parent_i<roulette_wheel.size()) && (random2>=roulette_wheel[selected_parent_i]); selected_parent_i++);
	clp2 = action_set[selected_parent_i];	// to be changed if list containers are used
	assert(selected_parent_i<roulette_wheel.size());

	assert(clp1->id()==(*test1)->id());
	assert(clp2->id()==(*test2)->id());
}

void	
xcsf_classifier_system::create_prediction_array()
{
	t_action		action;
	t_system_prediction	prediction;

	//! clear the prediction array
	prediction_array.clear();

	//!	build the prediction array with all the possible actions
	action.reset_action();

	do {
		prediction.action = action;
		prediction.n = 0;
		prediction.payoff = 0;
		prediction.sum = 0;
		prediction_array.push_back(prediction);
	} while (action.next_action());
}

void	
xcsf_classifier_system::init_prediction_array()
{
	vector<t_system_prediction>::iterator	sp;

	for(sp=prediction_array.begin(); sp!=prediction_array.end(); sp++)
	{
		sp->n =0;
		sp->payoff = 0;
		sp->sum = 0;
	}
}



//! delete a set of classifiers from [P], [M], [A], [A]-1
void	
xcsf_classifier_system::as_subsume(t_set_iterator classifier, t_classifier_set &set)
{
	t_set_iterator	sp;		//! iterator for visiting the set of classifier
	t_set_iterator	pp;		//! iterator for visiting [P]

        t_classifier *most_general;	//! keeps track of the most general classifier

	most_general = *classifier;

	for(sp=set.begin(); sp!=set.end(); sp++)
	{
		//! remove cl from [M], [A], and [A]-1
		t_set_iterator	clp;
		clp = find(action_set.begin(),action_set.end(), *sp);
		if (clp!=action_set.end())
		{
			action_set.erase(clp);
		}

		clp = find(match_set.begin(),match_set.end(), *sp);
		if (clp!=match_set.end())
		{
			match_set.erase(clp);
		}

		clp = find(previous_action_set.begin(),previous_action_set.end(), *sp);
		if (clp!=previous_action_set.end())
		{
			previous_action_set.erase(clp);
		}


		pp = lower_bound(population.begin(),population.end(),*sp,compare_cl);
		if ((pp!=population.end()))
		{
			//! found another classifier, something is wrong
			if ( (*pp)!=(*sp) )
			{
				xcs_utility::error(class_name(),"as_subsumption", "classifier not found", 1);
				exit(-1);
			}
		}

		macro_size--;

                most_general->numerosity += (*pp)->numerosity;

		delete *pp;
		population.erase(pp);
	}
	set.clear();
}

//! find the classifiers in set that are subsumed by the classifier 
void	
xcsf_classifier_system::find_as_subsumed(
	t_set_iterator classifier, 
	t_classifier_set &set, 
	t_classifier_set &subsumed)
const
{
	// t_set_iterator	sp;	//! pointer to the elements in the classifier set

	subsumed.clear();

	if (classifier!=set.end())
	{
		for(t_set_const_iterator sp=set.begin(); sp!=set.end(); sp++ )
		{
			if (*classifier!=*sp)
			{
				if ((*classifier)->is_more_general_than(**sp))
				{
					subsumed.push_back(*sp);
				}
			}
		}
	}
}

//! perform action set subsumption on the classifier in the set
void	
xcsf_classifier_system::do_as_subsumption(t_classifier_set &set)
{
	/*! 
	 * \brief check whether the condition type allow action set subsumption
	 *
	 * the same check is already performed at construction time. 
	 * thus this might be deleted.
	 */
	t_condition	cond;
	if (!cond.allow_as_subsumption())
	{
		xcs_utility::error(class_name(),
			"do_as_subsumption", 
			"condition does not allow action set subsumption", 1);
	}

	//! find the most general classifier
	t_set_iterator most_general;
	
	most_general = find_most_general(set);

	if ((most_general!=set.end()) && !classifier_could_subsume(**most_general, epsilon_zero, theta_sub))
	{
		xcs_utility::error(class_name(),
			"do_as_subsumption", 
			"classifier could not subsume", 1);
	}


	//! if there is a "most general" classifier, it extracts all the subsumed classifiers
	if (most_general!=set.end())
	{
		t_classifier_set subsumed;

		find_as_subsumed(most_general, set, subsumed);

		if (subsumed.size())
		{
			as_subsume(most_general, subsumed);
		}
	}
}

xcsf_classifier_system::t_set_iterator
xcsf_classifier_system::find_most_general(t_classifier_set &set) const
{
	t_set_iterator 	most_general = set.end();	
	
	t_set_iterator	sp;

	for( sp=set.begin(); sp!=set.end(); sp++ )
	{
		if ( classifier_could_subsume( (**sp), epsilon_zero, theta_as_sub) )
		{
			if (most_general==set.end())
				most_general = sp;
			else {
				if ( (*sp)->subsume(**most_general))
						
					most_general = sp;
			}
		}
	}

	return most_general;
}

void
xcsf_classifier_system::init_population_random()
{
	unsigned long	cl;
	xcslib::timer		check;

	clear_population();

	check.start();
	for(cl=0; cl<max_population_size; cl++)
	{
		t_classifier classifier;
		classifier.random();
		init_classifier(classifier);
		insert_classifier(classifier);
	}

	check.stop();
	macro_size = population.size();
	population_size = max_population_size;
};

void
xcsf_classifier_system::print_prediction_array(ostream& output) const
{
	vector<t_system_prediction>::const_iterator		pr;	

	for(pr=prediction_array.begin(); pr!=prediction_array.end(); pr++)
	{
		output << "(" << pr->action << ", " << pr->payoff << ")";
	}
}

void
xcsf_classifier_system::tournament_selection(t_classifier_set& set, t_classifier_ptr& clp, double tournament_size)
{
	t_set_iterator	as;	
	t_set_iterator	winner = set.end();

	while (winner==set.end())
	{
		for(t_set_iterator as=set.begin(); as!=set.end(); as++)
		{
			bool selected = false;

			for(unsigned long num=0; (!selected && (num<(**as).numerosity)); num++)
			{
				if (xcs_random::random()<tournament_size)
				{
					if ((winner==set.end()) ||
					    (((**winner).fitness/(**winner).numerosity)<((**as).fitness/(**as).numerosity)))
					{
						winner = as;
						selected = true;
					}
				}
			}
		}
	}

	clp = *winner;
}

void
xcsf_classifier_system::init_population_load(string filename)
{

	population.clear();		//! clear [P] before loading (20030808)

	ifstream	POPULATION(filename.c_str());

	if (!POPULATION.good())
	{
		xcs_utility::error( class_name(), "init_population_load", "file <"+filename+"> not found", 1);
	}
	t_classifier	in_classifier;
	unsigned long	n = 0;
	macro_size = 0;
	population_size = 0;

	string str_classifier;
	while(!POPULATION.eof())
	{
		std::getline(POPULATION, str_classifier);
		t_classifier classifier;

		str_classifier = xcs_utility::trim(str_classifier);
		cout << "STR  " << str_classifier << endl;

		if (str_classifier!="")
		{
			stringstream CLASSIFIER(str_classifier);
			CLASSIFIER >> in_classifier;
			cout << "READ " << in_classifier << endl << endl;

			t_classifier	*classifier = new t_classifier(in_classifier);
			classifier->time_stamp = total_steps;
			population.push_back(classifier);
			population_size += classifier->numerosity;
			macro_size++;
		}
	}

}

void
xcsf_classifier_system::init_population_solution(string filename)
{

	population.clear();		//! clear [P] before loading (20030808)

	ifstream	SOLUTION(filename.c_str());
	string		condition;
	string		action;
	double 		prediction;

	if (!SOLUTION.good())
	{
		xcs_utility::error( class_name(), "init_population_solution", "file <"+filename+"> not found", 1);
	}

	t_classifier	*in_classifier;
	unsigned long	n = 0;
	macro_size = 0;
	population_size = 0;

	while(!(SOLUTION.eof()) && SOLUTION>>condition>>action>>prediction)
	{
		t_classifier	*classifier = new t_classifier();

		classifier->condition.set_string_value(condition);
		classifier->action.set_string_value(action);
		classifier->prediction = prediction;
		population.push_back(classifier);
		macro_size++;
	}

	unsigned long numerosity = max_population_size/macro_size;

	population_size = 0;

	for(size_t i=0; i<population.size(); i++)
	{
		population[i]->fitness = 1.0;
		population[i]->error = 0.0;
		population[i]->actionset_size = numerosity;
		population[i]->numerosity = numerosity;
		population[i]->experience = 1;
		population[i]->time_stamp = total_steps;
		population_size = population_size+numerosity;
	}

	for(size_t i=0; i<population.size() && population_size<max_population_size; i++)
	{
		population[i]->numerosity++;
		population_size++;
	}

	for(size_t i=0; i<population.size(); i++)
	{
		cout << *(population[i]) << endl;
	}

}

xcsf_classifier_system::t_set_iterator
xcsf_classifier_system::select_classier_for_deletion(t_classifier_set &set)
{
	// compute the average fitness
	unsigned long size = 0;
	double average_fitness = 0.;
	for(t_set_iterator pp=set.begin(); pp!=set.end(); pp++)
	{
		average_fitness += (**pp).fitness;
		size += (**pp).numerosity;
	}

	average_fitness /= ((double) size);

	// crosscheck that computed values and maintained values are coherent
	assert(population_size==size);


	vector<double>	select;
	select.reserve(set.size());

	double		deletion_score_sum = 0;
	for(t_set_iterator pp=set.begin(); pp!=set.end(); pp++)
	{
		//! compute the deletion vote;
		double deletion_score = (**pp).actionset_size * (**pp).numerosity;
		double normalized_fitness = (**pp).fitness/double((**pp).numerosity);

		// the deletion score is increased when the classifier is experienced 
		// its fitness is low compared to the average fitness
		if (((**pp).experience>theta_del) && (normalized_fitness<delta_del*average_fitness))
		{
			deletion_score = deletion_score * average_fitness/normalized_fitness;
		}
		
		deletion_score_sum += deletion_score;
		select.push_back(deletion_score_sum);
	}

	double random = deletion_score_sum*(xcs_random::random());

	vector<double>::iterator pp_test = lower_bound(select.begin(),select.end(),random);

	size_t index = (pp_test - select.begin());

	return (population.begin()+index);
}



//! delete classifier from the population according to the selected strategy
void
xcsf_classifier_system::delete_classifier()
{	
	t_set_iterator 	pp;

	double		average_fitness = 0.;
	double		vote_sum;
	double		vote;
	double		random;

	unsigned long	sel;

	if (population_size<=max_population_size)
	{
		return;
	}

	pp = select_classier_for_deletion(population);

	if ((**pp).numerosity>1)
	{
		(**pp).numerosity--;
		population_size--;
		return;
	}

	// it is the last copy of the classifier so delete it from everywhere [A], [A]-1, [M], and [P]
	t_set_iterator	clp;
	clp = find(action_set.begin(),action_set.end(), *pp);
	if (clp!=action_set.end())
	{
		action_set.erase(clp);
	}

	clp = find(match_set.begin(),match_set.end(), *pp);
	if (clp!=match_set.end())
	{
		match_set.erase(clp);
	}

	clp = find(previous_action_set.begin(),previous_action_set.end(), *pp);
	if (clp!=previous_action_set.end())
	{
		previous_action_set.erase(clp);
	}

	delete *pp;
	
	population.erase(pp);
	population_size--;
	macro_size--;
}

//! check whether cl is subsumed by any of the classifiers in the action set
xcsf_classifier_system::t_set_iterator
xcsf_classifier_system::find_subsuming_classifier(const t_classifier &offspring, t_classifier_set &action_set)
{
	t_set_iterator subsuming_classifier_iterator = action_set.end();
	
	//! set the time stamp of classifiers in [A]
	for(t_set_iterator as=action_set.begin(); (subsuming_classifier_iterator==action_set.end())&&(as!=action_set.end()); as++)
	{
		if (subsume(**as, offspring))
		{
			subsuming_classifier_iterator = as;	
		}
	}

	return subsuming_classifier_iterator;
}

xcsf_classifier_system::t_set_iterator 
xcsf_classifier_system::select_proportional(t_classifier_set& classifier_set, const vector<double>& select, double value) const
{
	vector<double>::const_iterator selected = lower_bound(select.begin(),select.end(),value);

	size_t index = (selected - select.begin());

	return classifier_set.begin()+index;
}

std::vector<double>
xcsf_classifier_system::predict(t_state inputs)
{
	unsigned long match_set_size;
	vector<double> prediction;

        vector<double> numerical_inputs = inputs.numeric_representation();

        do {
            match_set_size = match(inputs);
            build_prediction_array(numerical_inputs);
        } while (covering(inputs));

        //! build the prediction array P(.)
        build_prediction_array(numerical_inputs);

        for(int i=0; i<prediction_array.size(); i++)
	{
		prediction.push_back(0);
	}
	for(int i=0; i<prediction_array.size(); i++)
	{
		prediction[prediction_array[i].action.value()]=prediction_array[i].payoff;
	}

	return prediction;
}