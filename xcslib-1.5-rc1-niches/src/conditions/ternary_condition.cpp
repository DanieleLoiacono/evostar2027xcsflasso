#include <string>
#include <fstream>
#include <algorithm>
#include "xcs_definitions.h"
#include "xcs_random.h"
// #include "xcs_configuration_manager.h"
#include "condition_base.h"
#include "ternary_condition.h"

using namespace std;

bool			ternary_condition::init = false;	
unsigned long	ternary_condition::no_bits;
double			ternary_condition::dont_care_prob;
unsigned long	ternary_condition::crossover_type;
unsigned long	ternary_condition::mutation_type;
bool			ternary_condition::flag_mutation_with_dontcare;

const std::vector<std::string> ternary_condition::configuration_parameters = {"condition size", "dontcare probability", "mutate with dontcare","crossover","mutation"};

const char	dont_care = '#';

ternary_condition::ternary_condition()
{
	if (!ternary_condition::init)
	{
		xcs_utility::error(class_name(),"ternary_condition()", "not inited", 1);
	}
}

ternary_condition::ternary_condition(xcslib::configuration_manager& xcs_config)
{
	string		str_mutation_with_dontcare;
	ifstream 	config;
	
	if (!ternary_condition::init)
	{
		if (!xcs_config.exist(tag_name()))
		{
			xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
		}
		
		xcs_config.check_parameters(tag_name(), configuration_parameters);

		set_parameters(xcs_config);
	}
	init = true;
};

void
ternary_condition::set_parameters(xcslib::configuration_manager& xcs_config)
{
	string str_crossover;
	string str_mutation;

	try {
		no_bits = xcs_config.Value(tag_name(), "condition size");
	} catch (...) {
		xcs_utility::error(class_name(), "constructor", "attribute \'condition size\' not found in <" + tag_name() + ">", 1);
	}

	try {
		dont_care_prob = xcs_config.Value(tag_name(), "dontcare probability");
	} catch (...) {
		xcs_utility::error(class_name(), "constructor", "attribute \'dontcare probability\' not found in <" + tag_name() + ">", 1);
	}

	xcs_utility::set_flag(xcs_config.Value(tag_name(), "mutate with dontcare", "on"), ternary_condition::flag_mutation_with_dontcare);

	try {
		str_crossover = (string) xcs_config.Value(tag_name(), "crossover");
	} catch (...) {
		xcs_utility::error(class_name(), "constructor", "attribute \'crossover\' not found in <" + tag_name() + ">", 1);
	}

	if (str_crossover=="one-point")
	{
		crossover_type = 1;
	} else if (str_crossover=="two-point") 
	{
		crossover_type = 2;
	} else if (str_crossover=="uniform") 
	{
		crossover_type = 0;
	} else {
		xcs_utility::error(class_name(), "constructor", "crossover method \'"+str_crossover+"\' not supported in <" + tag_name() + ">", 1);
	}

	str_mutation = (string) xcs_config.Value(tag_name(), "mutation", "input-based");
	if (str_mutation=="random")
	{
		xcs_utility::warning(class_name(), "set_parameters", "deprecated random mutation enabled");
		mutation_type = 0;
	} else if (str_mutation=="input-based")
	{
		mutation_type = 1;
	} else {
		xcs_utility::error(class_name(), "constructor", "mutation method \'"+str_mutation+"\' not supported in <" + tag_name() + ">", 1);
	}
}

void
ternary_condition::print_parameters(ostream& OUTPUT)
const
{
	OUTPUT << "<" << tag_name() << ">" << endl;
	OUTPUT << "\t" << "condition size = " << no_bits << endl;
	OUTPUT << "\t" << "dontcare probability" << " = " << dont_care_prob << endl;
	OUTPUT << "\t" << "mutate with dontcare" << " = " << (ternary_condition::flag_mutation_with_dontcare?"yes":"no") << endl;

	vector<string> str_crossover_methods = {"uniform","one-point","two-point"};
	OUTPUT << "\t" << "crossover" << " = " << str_crossover_methods[crossover_type] << endl;

	vector<string> str_mutation_methods = {"random","input-based"};
	OUTPUT << "\t" << "mutation" << " = " << str_mutation_methods[mutation_type] << endl;

	OUTPUT << "</" << tag_name() << ">" << endl;
}

ternary_condition::~ternary_condition()
{

}

bool 
ternary_condition::operator<(const ternary_condition& cond) const
{
	return (bitstring<cond.string_value());
};

bool 
ternary_condition::operator==(const ternary_condition& cond) const
{
	return (bitstring==cond.string_value());
};

bool 
ternary_condition::operator!=(const ternary_condition& cond) const
{
	return (bitstring!=cond.string_value());
}

/*!
 *	assignment operators
 */
// ternary_condition&
// ternary_condition::operator=(ternary_condition& cond)
// {
// 	bitstring = cond.string_value();
// 	return (*this);
// }

ternary_condition&
ternary_condition::operator=(const ternary_condition& cond)
{
	bitstring = cond.string_value();
	return (*this);
}

// 
// 	match operator
//
bool
ternary_condition::match(const binary_inputs& sens) const
{
	string::size_type	bit;
	string			input;
	bool			result;

	input = sens.string_value();

	assert(input.size()==bitstring.size());

	bit = 0;
	result = true;

	while ( (result) && (bit<bitstring.size()) )
	{
		result = ( (bitstring[bit]=='#') || (bitstring[bit]==input[bit]) );
		bit++;
	}
	
	return result;
}

// 
// 	cover operator
//
void
ternary_condition::cover(const binary_inputs& sens) 
{
	string::size_type	bit;
	string			input;

	input = sens.string_value();

	bitstring = "";

	for(bit = 0; bit<input.size(); bit++)
	{
		if (xcs_random::random()<ternary_condition::dont_care_prob)
		{
			//bitstring += ternary_condition::dont_care;
			bitstring += '#';
		} else {
			bitstring += input[bit];
		}
	}
}

// 
// 	mutate operator
//
void
ternary_condition::mutate(double mutation_rate, const binary_inputs &inputs)
{
	//! if mutation==1, restricted (1-value) mutation is required
	if (mutation_type==1)
	{
		string str_inputs = inputs.string_value();
	
		assert(str_inputs.size()==bitstring.size());
	
		for(unsigned long bit = 0; bit<bitstring.size(); bit++)
		{
			if (xcs_random::random()<mutation_rate)
			{
				if (bitstring[bit]=='#')
				{
					bitstring[bit] = str_inputs[bit];
				} else {
					if (flag_mutation_with_dontcare)
						bitstring[bit] = '#';
				}
			}
		}
	} else {
		mutate(mutation_rate);
	}
}

void
ternary_condition::mutate(double mutation_rate)
{
	string::size_type	bit;

	for(bit = 0; bit<bitstring.size(); bit++)
	{
		if (xcs_random::random()<mutation_rate)
		{
			if (bitstring[bit]=='#')
			{
				bitstring[bit] = '0'+xcs_random::dice(2);
			} else {
				if (flag_mutation_with_dontcare)
				{
					char values[2] = {'?', '#'};
					values[0] = ('1' - bitstring[bit])+'0';
					bitstring[bit] = values[xcs_random::dice(2)];
				} else {
					bitstring[bit] = ('1' - bitstring[bit])+'0';
				}
			}
		}
	}
}

void 
ternary_condition::recombine(ternary_condition& offspring, unsigned long method)
{
	switch(method)
	{
		case 0:
			uniform_crossover(offspring);
			break;
		case 1:
			single_point_crossover(offspring);
			break;
		case 2:
			two_point_crossover(offspring);
			break;
		default:
			xcs_utility::error(class_name(),"recombine","crossover method ("+std::to_string(method)+") not supported",1);
	}
}

bool 
ternary_condition::is_more_general_than(const ternary_condition& condition) 
const
{
	string str = condition.string_value();
	unsigned long sz = bitstring.size();

	unsigned long bit = 0;
	bool does_subsume = true;
	
	while  ( (bit<sz) && (does_subsume) )
	{
		does_subsume = ((bitstring[bit]=='#') || (str[bit]==bitstring[bit]));
		bit++;
	}

	return does_subsume;
}

void
ternary_condition::random() 
{
	string::size_type	bit;

	bitstring = "";

	for(bit = 0; bit<ternary_condition::no_bits; bit++)
	{
		if (xcs_random::random()<ternary_condition::dont_care_prob)
		{
			bitstring += '#';
		} else {
			bitstring += '0' + xcs_random::dice(2);
		}
	}
}

/// single point crossover
void	
ternary_condition::single_point_crossover(ternary_condition& offspring)
{
        unsigned long sz = size();
        unsigned long point = 1+xcs_random::dice(sz-1);
	string str = offspring.string_value();

	for(unsigned long bit=point; bit<sz; bit++)
	{
		swap(bitstring[bit],str[bit]);
	}
	offspring.set_string_value(str);
}

//! two point crossover
void	
ternary_condition::two_point_crossover(ternary_condition& offspring)
{
	unsigned long	bit;
	string		str = offspring.string_value();

	unsigned long	x;
	unsigned long	y;

	x = xcs_random::dice(offspring.size()+1);
	y = xcs_random::dice(size()+1);
	if (x>y)
		swap(x,y);
	bit = 0;

	do {
		if ((x<bit) && (bit<y))
			swap(bitstring[bit], str[bit]);
		bit++;
	} while ( bit<y );

	offspring.set_string_value(str);

}

//! uniform crossover
void	
ternary_condition::uniform_crossover(ternary_condition& offspring)
{
        unsigned long sz = size();

	string str = offspring.string_value();

	for(unsigned long bit=0; bit<sz; bit++)
	{
		if (xcs_random::random()<.5)
			swap(bitstring[bit],str[bit]);
	}
	offspring.set_string_value(str);
}

//! generality
double 
ternary_condition::generality() const
{
	return 1-specificity();
};

//! specificity
double 
ternary_condition::specificity() const
{
	string::const_iterator	ch;
	
	double specificity = 0;
	
	for( ch=bitstring.begin(); ch!=bitstring.end(); ch++ )
	{
		specificity += (*ch!='#')?1:0;	
	}

	specificity = specificity/bitstring.size();

	return specificity;
};
