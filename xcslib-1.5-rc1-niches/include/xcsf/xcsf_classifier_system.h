#include <list>
#include "xcs_definitions.h"
#include "xcs_random.h"
#include "xcs_statistics.h"
#include "configuration_manager.h"

using namespace std;

#ifndef __xcsf_classifier_system__
#define __xcsf_classifier_system__

/*!
 * \class xcsf_classifier_system
 * \brief implements the XCS classifier system
 */
class xcsf_classifier_system
{
public:
    //================================================================================
    //
    //
    //	PUBLIC TYPES
    //
    //
    //================================================================================

    //! pointer to a classifier
    typedef t_classifier *t_classifier_ptr;

    //! set of classifiers
    /*!
     * \type t_classifier_set
     * \brief represent a set of classifiers
     */
    typedef vector<t_classifier *> t_classifier_set;

    //! index in set of classifiers
    /*!
     * \type t_set_iterator
     * \brief represent an iterator over a set of classifiers
     */
    typedef vector<t_classifier *>::iterator t_set_iterator;
    typedef vector<t_classifier *>::const_iterator t_set_const_iterator;

    //! selection strategy
    enum class offspring_selection {
        roulette_wheel,       //! select offspring classifiers based on RWS and fitness
        tournament_selection, //! offspring selection based on TS and fitness maximization
    };

    //! action selection strategy
    enum class action_selection {
        greedy,         // => best action
        random,         // => random action
        epsilon_greedy, // => random with probability epsilon, best action otherwise
    };

    //! population init strategy
    enum class population_initialization {
        random,  // random population
        empty,   // empty population
        load,    // saved population
        solution // simple solution with condition-action-payoff
    };

    /*!
     * \brief implements the elements of the prediction array
     */
    class t_system_prediction {
      public:
        double payoff;   //! action payoff
        t_action action; //! action
        double sum;      //! fitness sum to normalize action payoff
        unsigned long n; //! number of classifiers that advocate the action

        t_system_prediction() : payoff(0), sum(0), n(0) {};

        // used to find the prediction of a given action
        int operator==(const t_action &action) const { return (this->action == action); }
    };

  public:
    //! name of the class that implements XCS
    string class_name() const { return string("xcsf_classifier_system"); };

    //! tag used to access the configuration file
    string tag_name() const { return string("classifier_system"); };

    //--------------------------------------------------------------------------------
    // Class constructor and parameters printing/setting
    //--------------------------------------------------------------------------------

  public:
    xcsf_classifier_system(xcslib::configuration_manager &xcs_config, t_environment *environment);
    void print_parameters(ostream &output) const;

  private:
    void set_parameters(xcslib::configuration_manager &xcs_config);
    void set_init_strategy(string);
    void set_exploration_strategy(const string &);
    void set_offspring_selection_strategy(const string &, offspring_selection &selection_strategy,
                                          double &tournament_size);

    const static std::vector<std::string> configuration_parameters;

  public:
    //--------------------------------------------------------------------------------
    // Interface methods for the experimental setup
    //--------------------------------------------------------------------------------

    //! define what has to be done when an experiment begins/ends
    void begin_experiment();
    void end_experiment();

    //! define what has to be done when an new problem begins/ends.
    void begin_problem();
    void end_problem();

    //!	performs one full cycle 
    void step(const bool exploration_mode, const bool condensationMode);

    //! return the number of macro and micro classifiers
    unsigned long no_macroclassifiers() const { return population.size(); };

    //! Export a read-only population view; ownership stays with the system.
    vector<const t_classifier*> population_view() const {
        return vector<const t_classifier*>(population.begin(), population.end());
    }
    unsigned long get_population_size() const { return population_size; };

    //! return the current system error
    double get_system_error() const { return system_error; };

	//! XCS will update the parameters during testing
    bool update_during_test_problems() const { return flag_update_test; }

    //--------------------------------------------------------------------------------
    // Methods to save information to files
    //--------------------------------------------------------------------------------

  public:

    //! writes trace information on an output stream;
    string trace() const { return "-"; };

    //! return the collected statistics for the current experiment
    xcs_statistics statistics() const { return stats; };

    //!	restore XCS state from an input stream
    void restore_state(istream &input);

    //!	save the experiment state to an output stream
    void save_state(ostream &output);

    //!	save population
    void save_population(ostream &ouput);

    //--------------------------------------------------------------------------------
    // Performance Component
    //--------------------------------------------------------------------------------

  private:

    bool init;                  //! true, if the class was initialized
    t_environment *environment; //! link to the environment

    t_classifier_set population;          //! population [P]
    t_classifier_set match_set;           //! match set [M]
    t_classifier_set action_set;          //! action set [A]
    t_classifier_set previous_action_set; //! action set at previous time step [A]-1

    t_state previous_input;							    //! input at t-1
    vector<double> previous_input_numeric;	//! numeric representation of input at t-1  
    t_state current_input;							    //! current input at time t
    
    vector<t_system_prediction> prediction_array; 	//! prediction array P(.)
    vector<unsigned long> available_actions; //! actions in the prediction array that have a not null prediction, and
                                             //! thus are available for selection

    double previous_reward; //! reward received at previous time step

    bool exploration_mode; //! true if the problem is solved in exploration (Wilson 1995), i.e., XCS is in learning
                           //! (Butz 2001)

    //! experiment parameters
    unsigned long total_steps;          //! total number of steps
    unsigned long total_learning_steps; //! total number of exploration steps
    unsigned long total_time;           //! total time passed (should be the same as total steps);
    unsigned long problem_steps;        //! total number of steps within the single problem
    double total_reward;                //! total reward gained during the problem
    double system_error; //! difference between predicted payoff and reward received (it is used only in single step
                         //! problems)

    //! [P] parameters
    unsigned long max_population_size;         //! maximum number of micro classifiers
    unsigned long population_size;             //! population size
    unsigned long macro_size;                  //! number of macroclassifiers
    population_initialization population_init; //! init strategy for [P]
    string population_init_file;               //! file containing a population to be used to init [P]

    //! classifier parameters
    double init_prediction;        //! initial prediction for newly created classifiers
    double init_error;             //! initial predition error for newly created classifiers
    double init_fitness;           //! initial fitness for newly created classifiers
    double init_set_size;          //! initial set size for newly created classifiers
    unsigned long init_no_updates; //! initial number of updates for newly created classifiers

    //! covering
    double tetha_nma; //! minimum number of actions in [M]

    //! action selection
    action_selection action_selection_strategy; //! strategy for action selection
    double prob_random_action;                  //! probability of random action

    //--------------------------------------------------------------------------------
    // Reinforcement Component
    //--------------------------------------------------------------------------------

    double learning_rate;   		//! beta parameter
    double discount_factor; 		//! gamma parameter, the discount factor
    bool flag_update_test;			//! true if update is performed during test
    bool flag_use_mam;				//! use MAM for the initial updates
    bool flag_use_gradient_descent; //! true if gradient descent is used

    //--------------------------------------------------------------------------------
    // Discovery Component
    //--------------------------------------------------------------------------------

    double epsilon_zero; //! epsilon zero parameter, determines the threshold for accurate classifiers
    double alpha;        //! alpha parameter, determines the start of the decay
    double vi;           //! vi parameter, determines the decay rate

    offspring_selection ga_offspring_selection_strategy;
    double ga_tournament_size;

    offspring_selection condensation_offspring_selection_strategy;
    double condensation_tournament_size;

    bool flag_discovery_component;	//! true if the GA will be used
    double theta_ga;				//! threshold for GA activation
    double prob_crossover;			//! probability to apply crossover
    double prob_mutation;			//! probability to apply mutation
    bool flag_error_update_first; 	//! if true, prediction error is updated first

    bool use_ga;        //! true if GA is on
    bool use_crossover; //! true if crosseover is used
    bool use_mutation;  //! true if mutation is used

    //! subsumption deletion parameters
    bool flag_ga_subsumption;  //! true if GA subsumption is on
    bool flag_gaa_subsumption; //! true if Butz GA subsumption in [A] is used (can be active only if standard subsumption is active)
    bool flag_as_subsumption;  //! true if AS subsumption is on
    double theta_sub;          //! threshold for subsumption activation
    double theta_as_sub;       //! threshold for subsumption activation
    // bool			flag_cover_average_init;

    //! delete parameters
    bool flag_delete_with_accuracy; //! deletion strategy: true, if T3 is used; false if T1 is used
    double theta_del;               //! theta_del parameter for deletion
    double delta_del;               //! delta parameter for deletion

    //--------------------------------------------------------------------------------
    // Additional fields
    //--------------------------------------------------------------------------------

    xcs_statistics stats; //! classifier system statistics
    vector<double> select; //! vector for roulette wheel selection
    vector<double> error;

    //--------------------------------------------------------------------------------
    // Niche identification and tracking
    //--------------------------------------------------------------------------------

#ifdef __NICHE_TRACKING__
    unsigned long max_niche_queue_size;
#endif

#ifdef __TRACK_ACTION_SETS__
private:
    vector<string> saved_action_sets;

    void track_action_set(const t_classifier_set &action_set)
    {
      string str_action_set = std::to_string(action_set[0]->fitness)+";"+std::to_string(action_set[0]->numerosity);

      for(size_t action_set_i=1; action_set_i<action_set.size(); action_set_i++)
      {
        str_action_set += "|" + std::to_string(action_set[action_set_i]->fitness)+";"+std::to_string(action_set[action_set_i]->numerosity);
      }

      saved_action_sets.push_back(str_action_set);
    }
public:
    void save_as_data(ostream &output) const
    {        
        output << "Action Set" << endl;
        for(size_t as_data_i=0; as_data_i<saved_action_sets.size(); as_data_i++)
            output << saved_action_sets[as_data_i] << endl;
    }
#endif

#ifdef __COMPARE_TS__
    //--------------------------------------------------------------------------------
    // Fast Tournament Selection
    //--------------------------------------------------------------------------------
private:
    struct ts_data 
    {
      public:
        ts_data(
          double time_, 
          unsigned long ticks_, 
          double time_macro_, 
          unsigned long ticks_macro_, 
          unsigned long no_classifiers_, unsigned long as_size_, string action_set_, bool condensation_):
            time(time_),
            ticks(ticks_),
            time_macro(time_macro_),
            ticks_macro(ticks_macro_),
            no_classifiers(no_classifiers_),
            as_size(as_size_),
            action_set(action_set_),
            condensation(condensation_) {}

        double time;
        unsigned long ticks;

        double time_macro;
        unsigned long ticks_macro;

        unsigned long no_classifiers;
        unsigned long as_size;

        bool condensation;

        string action_set;
        friend std::ostream& operator<< (std::ostream& stream, const ts_data& data)
        {
          stream << data.time;
          stream << "," << data.ticks;
          stream << "," << data.time_macro;
          stream << "," << data.ticks_macro;
			    stream << "," << data.no_classifiers;
			    stream << "," << data.as_size;
			    stream << "," << data.action_set;
          stream << "," << data.condensation;

          return stream;
        }

        string to_string() const
        {
          ostringstream oss;
          oss << time;
          oss << "," << ticks;
          oss << time_macro;
          oss << "," << ticks_macro;
			    oss << "," << no_classifiers;
			    oss << "," << as_size;
			    oss << "," << action_set;
          oss << "," << condensation;
          return oss.str();
        }
    };

    
    public:
      vector<ts_data> ts_data_vector;
      void save_ts_data(ostream &output) const
      {
        output << "Time,Ticks,Time (Macro),Ticks (Macro),NoClassifiers,NoMicroClassifiers,ActionSet,Condensation" << endl;
        for(size_t data_i=0; data_i<ts_data_vector.size(); data_i++)
		      output << ts_data_vector[data_i] << endl;
      }

#endif

  private:
    //--------------------------------------------------------------------------------
    // Methods to manage the population
    //--------------------------------------------------------------------------------
    void clear_population();				//! empty [P]
    void init_classifier_set();				//! init [P] according to the selected strategy (i.e., empty or random)

    void init_population_random(); 				//! [P] is randomly generated
    void init_population_load(string);			//! [P] is loaded from a saved population
    void init_population_solution(string);      //! load a solution specified with condition-action-prediction (fitness,
                                                //! error and other parameters are initialized)

    t_set_const_iterator find_classifier(const t_classifier_set &population, const t_classifier &cs) const;

    void init_classifier(t_classifier &classifier); //! init the classifier parameters

    void insert_classifier(const t_classifier &);		//! insert a classifier in [P]

    void delete_classifier(); //! delete a classifier from [P]
    t_set_iterator select_classier_for_deletion(t_classifier_set &set);
    t_set_iterator select_proportional(t_classifier_set &, const vector<double> &, double) const;

    //--------------------------------------------------------------------------------
    // Performance Component Methods
    //--------------------------------------------------------------------------------

  public:
    //! print the prediction array P(.) to an output stream
    void print_prediction_array(ostream &) const;

	//! output the prediction vector for a given input. It is used to save the action-value-function.
    std::vector<double> predict(t_state inputs);	

  private:
    //!  build the match set [M]; it returns the number of microclassifiers that match the sensory configuration
    unsigned long match(const t_state &detectors);
    void create_prediction_array(); //! create the prediction array based on the action used
    void init_prediction_array(); //! clear the prediction array based
    void build_prediction_array(const vector<double>& current_input);  //!	build the prediction array P(.) from [M]
    bool covering(const t_state &); //! perform covering according to Butz and Wilson 2001
    t_action select_action(const action_selection) const; //! select an action based on the current policy
    t_action select_random_action() const;	//! select a random action
    t_action select_best_action() const;	//! select the best action
    void build_action_set(const t_action &); //! build the action set [A] for action

    //--------------------------------------------------------------------------------
    // Reinforcement Component Methods
    //--------------------------------------------------------------------------------

  private:
    void update_set(const double P, t_classifier_set &action_set,
                    const vector<double> &current_input); //! distributes the expected payoff to the classifiers

    //--------------------------------------------------------------------------------
    // Discovery Component Methods
    //--------------------------------------------------------------------------------

  private:

	//! update the values of classifiers' fitness
    void update_fitness(t_classifier_set &);

    //! updates the time stamps of the classifiers in the action set
    void update_timestamp(t_classifier_set &action_set);

    //! true -> enough time has passed from the last activation in this action set
    bool need_ga(t_classifier_set &action_set, const bool flag_explore);

    //! selects offspring classifiers using either roulette wheel or tournament selection
    void roulette_wheel_selection(t_classifier_set &action_set, t_classifier_ptr &clp1, t_classifier_ptr &clp2);
    void tournament_selection(t_classifier_set &, t_classifier_ptr &, double);
    void tournament_selection_macro(t_classifier_set& set, t_classifier_ptr& clp, double tournament_size);

    void compare_tournament_selection_methods(t_classifier_set &action_set, bool condensation);
    void select_offsprings(t_classifier_set &action_set, t_classifier_ptr &parent1, t_classifier_ptr &parent2);

    //! runs the genetic algorithm
    void genetic_algorithm(t_classifier_set &action_set, const t_state &detectors);

    //!	runs condensation
    void condensation(t_classifier_set &action_set);

    //! true -> if the offspring was subsumed by one of the parents or one of the classifiers in the action set
    bool apply_subsumption(const t_classifier &offspring, const t_classifier_ptr parent1, t_classifier_ptr parent2,
                           t_classifier_set &action_set);

    //--------------------------------------------------------------------------------
    // Subsumption in [A]: checks whether there is subsuming classifier in [A]
	  // it is used in Butz' XCS code when the parent does not subsume the offspring.
    //--------------------------------------------------------------------------------

    //! true -> if the classifier is subsumed by one classifier in the action set
    bool subsumed_by_the_classifier_set(const t_classifier &offspring, t_classifier_set &action_set);

    //! returns the position in the population of the classifier subsuming the offspring
    t_set_iterator find_subsuming_classifier(const t_classifier &offspring, t_classifier_set &action_set);

    //! true if classifier \emph first subsume classifier \emph second
    bool subsume(const t_classifier &first, const t_classifier &second);

    bool classifier_could_subsume(const t_classifier &classifier, double epsilon_zero, double theta_sub) const
    {
        return ((classifier.experience > theta_sub) && (classifier.error < epsilon_zero));
    };

    //--------------------------------------------------------------------------------
    // Action Set Subsumption from Wilson 1998 paper. It is never used in published
	// experiments since it is considered too strong. Subsumption in [A] from Butz
	// XCS C implementation is much better.
    //--------------------------------------------------------------------------------
    void do_as_subsumption(t_classifier_set &set); //! find the classifiers that are AS subsumed in the current set
    t_set_iterator
    find_most_general(t_classifier_set &) const; //!	find the "most general classifier in the set (Wilson 1998)

    //! find the classifiers in the set that are subsumed by classifier
    void find_as_subsumed(t_set_iterator, t_classifier_set &, t_classifier_set &) const;

    //!	perform AS subsumption, deleting the subsumed classifiers and increasing the numerosity of most general
    //! classifier
    void as_subsume(t_set_iterator, t_classifier_set &set);

    //--------------------------------------------------------------------------------
    // Utilities
    //--------------------------------------------------------------------------------
    void print_set(t_classifier_set &set, ostream &output); //! print a set of classifiers


};
#endif
