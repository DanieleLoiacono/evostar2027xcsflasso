/*!
 * \file bf_env.cc
 *
 * \brief implements the Boolean functions parity
 *
 * \author Pier Luca Lanzi
 *
 * \version 0.01
 *
 * \date 2005/08/01
 *
 */

// #define __DEBUG_ENVIRONMENT__

#include <sstream>
#include <cmath>
#include "xcs_utility.h"
#include "xcs_random.h"

#include "bf_env.h"

using namespace std;

//!< set the init flag to false so that the use of the config manager becomes mandatory
bool	bf_env::init=false;	

bf_env::bf_env(xcslib::configuration_manager& configuration)
{
	ifstream 	config;
	string 		str_input;
	string		str_function; 
	
	if (!bf_env::init)
	{
		bf_env::init = true;

		if (!configuration.exist(tag_name()))
		{
			xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
		}
		
		set_parameters(configuration);

		no_configurations = 1;
		no_configurations <<= state_size;
	}

	bf_env::init = true;
}

void
bf_env::set_parameters(xcslib::configuration_manager& xcs_config)
{
	string str_function;

	try {
		str_function = (string) xcs_config.Value(tag_name(), "function");
	} catch (const char *attribute) {
		xcs_utility::error(class_name(), "set_parameters", "attribute \'function\' not found in <" + tag_name() + ">", 1);
	}

	string str_noisy_reward = xcs_config.Value(tag_name(), "noisy reward", "off");

	if (xcs_utility::trim(str_noisy_reward)=="off")
	{
		flag_noisy_reward = false;
	} else {
		vector<string> values = xcs_utility::split(str_noisy_reward, " ");
		if (values.size()!=2)
		{
			xcs_utility::error(class_name(), "set_parameters","noisy reward needs two parameters (mean and standard deviation)",1);
		}

		reward_noise_mean = std::stod(xcs_utility::trim(values[0]));
		reward_noise_std = std::stod(xcs_utility::trim(values[1]));

		clog << "NOISY REWARD ACTIVE Mean = " << reward_noise_mean << " StdDev = " << reward_noise_std << endl;
		flag_noisy_reward = true;	
	}

	binary_function = get_binary_function(str_function);
	
	switch (binary_function)
	{
		case boolean_function::MULTIPLEXER:

			#ifdef __DEBUG_ENVIRONMENT__
			cout << "SETTING MULTIPLEXER PARAMETERS" << endl;
			#endif

			xcs_config.check_parameters(tag_name(), vector<string>({"function","address size","noisy reward"}));
			set_parameters_mp(xcs_config);
			break;

		case boolean_function::MAJORITY_ON:
		case boolean_function::CARRY:
			set_input_size_parameter(xcs_config);
			xcs_config.check_parameters(tag_name(), vector<string>({"function","input size","noisy reward"}));
			break;

		case boolean_function::EQUALITY:		
			xcs_config.check_parameters(tag_name(), vector<string>({"function","input size","number of ones","noisy reward"}));
			set_parameters_eq(xcs_config);
			break;
		default:
			xcs_utility::error(class_name(), "set_parameters","binary function not supported",1);
	}

	no_configurations = 1;
	no_configurations <<= state_size;
}

void
bf_env::print_parameters(ostream &OUTPUT)
const
{
	OUTPUT << "<" << tag_name() << ">" << endl;
	OUTPUT << "\t" << "function = " << get_binary_function_string(binary_function) << endl;

	switch (binary_function)
	{
		case boolean_function::MULTIPLEXER:
			OUTPUT << "\t" << "address size = " << address_size << endl;
			break;

		case boolean_function::MAJORITY_ON:
			OUTPUT << "\t" << "input size = " << state_size << endl;
			break;

		case boolean_function::CARRY:
			OUTPUT << "\t" << "input size = " << state_size << endl;
			break;

		case boolean_function::EQUALITY:		
			OUTPUT << "\t" << "input size = " << state_size << endl;
			OUTPUT << "\t" << "number of ones = " << no_ones << endl;
			break;
		default:
			xcs_utility::error(class_name(), "set_parameters","binary function not supported",1);
	}
	OUTPUT << "</" << tag_name() << ">" << endl;
}

void
bf_env::set_parameters_eq(xcslib::configuration_manager& xcs_config)
{
	try {
		state_size = (unsigned long) xcs_config.Value(tag_name(), "input size");
	} catch (...) {
		xcs_utility::error(class_name(), "constructor", "attribute \'input size\' not found in <" + tag_name() + ">", 1);
	}

	try {
		no_ones = (unsigned long) xcs_config.Value(tag_name(), "number of ones");
	} catch (...) {
		xcs_utility::error(class_name(), "constructor", "attribute \'number of ones\' not found in <" + tag_name() + ">", 1);
	}

	if ((no_ones<=0)||(no_ones>state_size))
	{
		xcs_utility::error(class_name(), "constructor", "attribute \'number of ones\' must be >0 <= input size", 1);
	}
}

void
bf_env::set_parameters_mp(xcslib::configuration_manager& xcs_config)
{
	try {
		address_size = (unsigned long) xcs_config.Value(tag_name(), "address size");
	} catch (...) {
		xcs_utility::error(class_name(), "constructor", "attribute \'address size\' not found in <" + tag_name() + ">", 1);
	}

	unsigned long BitStringSize = 1;
	BitStringSize <<= address_size;
	state_size = address_size + BitStringSize;
	state_size = address_size + long(pow(double(2),int(address_size)));

	// cout << "STATE SIZE = " << state_size << endl;

}
void
bf_env::set_input_size_parameter(xcslib::configuration_manager& xcs_config)
{
	try {
		state_size = (unsigned long) xcs_config.Value(tag_name(), "input size");
	} catch (...) {
		xcs_utility::error(class_name(), "constructor", "attribute \'input size\' not found in <" + tag_name() + ">", 1);
	}

	threshold = state_size/2;
}


bf_env::boolean_function 
bf_env::get_binary_function(string str_function)
{
	if (str_function=="multiplexer")
		return boolean_function::MULTIPLEXER;
	if (str_function=="equality")
		return boolean_function::EQUALITY;
	if (str_function=="majority")
		return boolean_function::MAJORITY_ON;
	if (str_function=="carry")
		return boolean_function::CARRY;

	xcs_utility::error(class_name(), "get_binary_function","binary function " + str_function + " not supported",1);	
}

string
bf_env::get_binary_function_string(boolean_function binary_function)
const
{
	switch (binary_function)
	{
		case boolean_function::MULTIPLEXER:
			return "multiplexer";

		case boolean_function::MAJORITY_ON:
			return "majority-on";

		case boolean_function::EQUALITY:
			return "eq";

		case boolean_function::CARRY:
			return "carry";

		default:
			xcs_utility::error(class_name(), "set_parameters","binary function not supported",1);
	}
}

/*!
 * \fn void bf_env::begin_problem(const bool explore)
 * \param explore true if the problem is solved in exploration
 *
 * \brief generates a new input configuration for the Boolean parity
 *
 * If the inputs must be visited uniformly, indicated by uniform_start set to true,
 * the variable current_configuration is used to generate the next available input;
 * otherwise, a random input configuration is generated
 */
void	
bf_env::begin_problem()
{
	string	str = "";

	for(string::size_type bit = 0; bit<state_size; bit++)
	{
		str += '0' + xcs_random::dice(2);
	}

	inputs.set_string_value(str);

	current_reward = 0;

	first_problem = false;
}

bool	
bf_env::stop()
const
{
	return(true); 
}

void	
bf_env::perform(const boolean_action& action)
{
	switch (binary_function)
	{
		case boolean_function::MULTIPLEXER:
			compute_mp(action);
			break;

		case boolean_function::EQUALITY:
			compute_eq(action);
			break;

		case boolean_function::MAJORITY_ON:
			compute_majority_on(action);
			break;

		case boolean_function::CARRY:
			compute_carry(action);
			break;

		default:
			xcs_utility::error(class_name(),"perform","function not supported",1);
	}

	if (flag_noisy_reward)
	{
		double noise = reward_noise_mean + xcs_random::nrandom()*reward_noise_std;
		current_reward += noise;
	}
}

//! only the current reward is traced
string
bf_env::trace() const
{
	return std::to_string(current_reward);
}

void 
bf_env::reset_input()
{
	current_state = 0;
	inputs.set_string_value(xcs_utility::long2binary(current_state,state_size));
}

bool 
bf_env::next_input()
{
	string	binary;
	bool	valid = false;

	current_state++; 
	if (current_state<no_configurations)
	{
		inputs.set_string_value(xcs_utility::long2binary(current_state,state_size));
		valid = true;
	} else {
		current_state = 0;
		inputs.set_string_value(xcs_utility::long2binary(current_state,state_size));
		valid = false;
	}
	return valid;
}

void
bf_env::save_state(ostream& output) const
{
	output << endl;
	output << current_state << endl;
}

void
bf_env::restore_state(istream& input)
{
	input >> current_state; 
	begin_problem();
}

bf_env::bf_env()
{
	if (!bf_env::init)
	{
		xcs_utility::error(class_name(),"class constructor", "not inited", 1);
	} else {
		// nothing to init
	}
}

void	
bf_env::compute_eq(const boolean_action& action)
{
	string				str_inputs;
	unsigned long		in;
	unsigned long		sum;
	unsigned long		result;
	
	str_inputs = inputs.string_value();

	sum = 0;

	for(in=0; in<inputs.size(); in++)
	{
		sum += str_inputs[in]-'0';
	}

	if (sum==no_ones)
		result = 1;
	else 
		result = 0;

	if (result==action.value())
	{
		current_reward = 1000;
	} else {
		current_reward = 0;
	}
}

void	
bf_env::compute_majority_on(const boolean_action& action)
{
	string				str_inputs;
	unsigned long		in;
	unsigned long		sum;
	unsigned long		result;
	
	str_inputs = inputs.string_value();

	sum = 0;

	for(in=0; in<inputs.size(); in++)
	{
		sum += str_inputs[in]-'0';
	}

	if (sum>int(str_inputs.size()/2))
	{
		result = 1;
	} else {
		result = 0;
	}

	if (result==action.value())
	{
		current_reward = 1000;
	} else {
		current_reward = 0;
	}
}

void	
bf_env::compute_mp(const boolean_action& action)
{

	#ifdef __DEBUG_ENVIRONMENT__
	cout << "compute_MP MULTIPLEXER PARAMETERS" << endl;
	#endif

	string str_inputs = inputs.string_value();

	unsigned long		address;
	unsigned long		index;		

	index = xcs_utility::binary2long(str_inputs.substr(0,address_size));
	address = address_size + index;

	if (!flag_layered_reward)
	{
		if ((str_inputs[address]-'0')==action.value())
		{
			current_reward = 1000;
		} else {
			current_reward = 0;
		}
	} else {
		if ((str_inputs[address]-'0')==action.value())
		{
			current_reward = 300 + index*200 + double(100*(unsigned long)(str_inputs[address]-'0'));
		} else {
			current_reward = index*200 + double(100*(unsigned long)(str_inputs[address]-'0'));
		}
	}
#ifdef __DEBUG_ENVIRONMENT__
	cout << "INPUT " << inputs << " BIT " << str_inputs[address] << " ACTION " << action.value() << " REWARD " << current_reward << endl;
#endif
}

void	
bf_env::compute_carry(const boolean_action& action)
{
	string				str_inputs;
	unsigned long		in;
	unsigned long		sum;
	unsigned long		result;
	
	str_inputs = inputs.string_value();

    unsigned int no_bits = str_inputs.size()/2;

    std::string first = str_inputs.substr(0,no_bits);
    std::string second = str_inputs.substr(no_bits,no_bits);

    int carry = 0;
    int digit = 0;

    for(int i=no_bits-1; i>=0; i--)
    {
        digit = (int(first[i]-'0')+int(second[i]-'0')+carry)%2;
        int updated_carry = (int(first[i]-'0')+int(second[i]-'0')+carry)/2;
        // std::cout << first[i] << "+" << second[i] << "+" << carry << " = " << digit << " carry " << updated_carry << std::endl;
        carry = updated_carry;
    }

	if (carry==action.value())
	{
		current_reward = 1000;
	} else {
		current_reward = 0;
	}
}


