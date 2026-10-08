#ifndef __BF_ENV__
#define __BF_ENV__

#include <vector>
#include <cassert>
#include <fstream>

#include "environment_base.h"
#include "configuration_manager.h"
#include "binary_inputs.h"
#include "boolean_action.h"

using namespace std;

class bf_env : public virtual environment_base<binary_inputs, boolean_action>
{
 public:

	enum class boolean_function {
		MULTIPLEXER, 
		EQUALITY,
		MAJORITY_ON,
		CARRY
	};

	string class_name() const { return string("bf_env"); };
	string tag_name() const { return string("environment::binary_function"); };
		
	//! Constructor for the parity class that read the class parameters through the configuration manager
	/*!
	 *  This is the first constructor that must be used. Otherwise an error is returned.
	 */
	bf_env(xcslib::configuration_manager&);
	
	void print_parameters(ostream &OUTPUT) const;


	//! Default constructor for the Boolean parity class
	/*!
	 *  If the class parameters have not yet initialized through the configuration manager, 
	 *  an error is returned and the method exists to shell. 
	 *  \sa bf_env(xcs_config_mgr&)
	 *  \sa xcs_config_mgr
	 */
	bf_env();
	
	void begin_experiment() {};
	void end_experiment() {};

	void begin_problem();
	void end_problem() {};

	bool stop() const;
	
	void perform(const boolean_action& action);

	string trace() const;

	void reset_input();
	bool next_input();

	void save_state(ostream& output) const;
	void restore_state(istream& input);
	
	virtual double reward() const {assert(current_reward==bf_env::current_reward); return current_reward;};

	virtual binary_inputs state() const { return inputs; };

	virtual bool is_terminal(const binary_inputs &state) const { return true; }

	virtual bool is_single_step() const {return true;}

	//! allow generating all inputs only for functions up to 11 bits (2048 configurations)
	virtual bool allow_test() const { return state_size<=8; }

 private:
	/*! \var bool init 
	 *  \brief true if the class parameters have been already initialized
	 */
	static bool init;

	/*! 
	 * \var binary_inputs inputs
	 * \brief inputs current input configuration
	 */
	binary_inputs inputs;			// input configuration

	/*! 
	 * \var bool first_problem 
	 * \brief true if the first problem is running
	 */
	bool first_problem;

	/*! \var unsigned long string_size
	 * \brief number of string bits for the Boolean parity
	 */
	unsigned long string_size;

	//! selected variables
	std::vector<unsigned long> selected_inputs;

	//! true if it the system must visit all the available configuration as a sequence
	bool uniform_start;		

	//! size of the whole parity string, e.g., 6 for the 6-way parity
	unsigned long state_size;

	//! the reward returned as a consequence of the last performed action
	double current_reward;

	//! the value that defines the equality function
	// unsigned long			k;

	void set_function(string str_function);
	
 private:
	void set_parameters(xcslib::configuration_manager&);
	void set_parameters_eq(xcslib::configuration_manager&);	
	void set_parameters_mp(xcslib::configuration_manager&);
	void set_input_size_parameter(xcslib::configuration_manager&);

	//! read selected variables
	void read_selected_variables(char*, std::vector<unsigned long>&) const;

	void compute_eq(const boolean_action& action);	
	void compute_mp(const boolean_action& action);
	void compute_majority_on(const boolean_action& action);
	void compute_carry(const boolean_action& action);

	boolean_function get_binary_function(string);
	string get_binary_function_string(boolean_function) const;

	boolean_function binary_function;

	// multiplexer parameters
	unsigned long address_size;
	bool flag_layered_reward = false;
	bool flag_noisy_reward = false;

	double reward_noise_mean;
	double reward_noise_std;

	// eq parameters
	unsigned long no_ones;

	// majority on parameters
	unsigned long threshold;

};
#endif
