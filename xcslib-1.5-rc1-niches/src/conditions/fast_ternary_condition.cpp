#include <string>
#include <fstream>
#include <algorithm>
#include "xcs_definitions.h"
#include "xcs_random.h"
// #include "xcs_configuration_manager.h"
#include "condition_base.h"
#include "fast_ternary_condition.h"

using namespace std;

//
//	condition parameters
//
bool		fast_ternary_condition::init = false;	
unsigned long	fast_ternary_condition::no_bits;
double		fast_ternary_condition::dont_care_prob;
unsigned long	fast_ternary_condition::crossover_type;
unsigned long	fast_ternary_condition::mutation_type;
bool		fast_ternary_condition::flag_mutation_with_dontcare;
const std::vector<std::string> fast_ternary_condition::configuration_parameters = {"condition size", "dontcare probability", "mutate with dontcare","crossover","mutation"};

const char	dont_care = '#';

fast_ternary_condition::fast_ternary_condition()
{
	if (!fast_ternary_condition::init)
	{
		xcs_utility::error(class_name(),"fast_ternary_condition()", "not inited", 1);
	}

}

fast_ternary_condition::fast_ternary_condition(xcs_configuration_manager& xcs_config)
{
	string		str_mutation_with_dontcare;
	ifstream 	config;
	
	if (!fast_ternary_condition::init)
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
fast_ternary_condition::set_parameters(xcs_configuration_manager& xcs_config)
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

	xcs_utility::set_flag(xcs_config.Value(tag_name(), "mutate with dontcare", "on"), fast_ternary_condition::flag_mutation_with_dontcare);

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
		mutation_type = 0;
	} else if (str_mutation=="input-based")
	{
		mutation_type = 0;
	} else {
		xcs_utility::error(class_name(), "constructor", "mutation method \'"+str_mutation+"\' not supported in <" + tag_name() + ">", 1);
	}
}

fast_ternary_condition::~fast_ternary_condition()
{

}

bool 
fast_ternary_condition::operator<(const fast_ternary_condition& cond) const
{
	return (bitstring<cond.string_value());
};

bool 
fast_ternary_condition::operator==(const fast_ternary_condition& cond) const
{
	return (bitstring==cond.string_value());
};

bool 
fast_ternary_condition::operator!=(const fast_ternary_condition& cond) const
{
	return (bitstring!=cond.string_value());
}

/*!
 *	assignment operators
 */
// fast_ternary_condition&
// fast_ternary_condition::operator=(fast_ternary_condition& cond)
// {
// 	bitstring = cond.string_value();
// 	return (*this);
// }

fast_ternary_condition&
fast_ternary_condition::operator=(const fast_ternary_condition& cond)
{
	bitstring = cond.string_value();
	
#ifdef __FAST_BINARY_MATCHING__
	set_match_vector(bitstring);
#endif
	return (*this);
}

// 
// 	match operator
//
bool
fast_ternary_condition::match(const binary_inputs& sens) const
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
fast_ternary_condition::cover(const binary_inputs& sens) 
{
	string::size_type	bit;
	string			input;

	input = sens.string_value();

	bitstring = "";

	for(bit = 0; bit<input.size(); bit++)
	{
		if (xcs_random::random()<fast_ternary_condition::dont_care_prob)
		{
			//bitstring += fast_ternary_condition::dont_care;
			bitstring += '#';
		} else {
			bitstring += input[bit];
		}
	}

#ifdef __FAST_BINARY_MATCHING__
	set_match_vector(bitstring);
#endif

}

// 
// 	mutate operator
//
// void
// fast_ternary_condition::mutate(const binary_inputs &sens, const double& mu)
void 
fast_ternary_condition::mutate(double mutation_rate, const binary_inputs &inputs)
{
	//! if mutation==1, restricted (1-value) mutation is required
	if (mutation_type==1)
	{	
		string input = inputs.string_value();

		assert(input.size()==bitstring.size());

		for(string::size_type bit = 0; bit<bitstring.size(); bit++)
		{
			if (xcs_random::random()<mutation_rate)
			{
				if (bitstring[bit]=='#')
				{
					bitstring[bit] = input[bit];
				} else {
					if (flag_mutation_with_dontcare)
						bitstring[bit] = '#';
				}
			}
		}
	} else {
		mutate(mutation_rate);
	}

	#ifdef __FAST_BINARY_MATCHING__
	set_match_vector(bitstring);
	#endif
}

void
fast_ternary_condition::mutate(double mutation_rate)
{
	for(string::size_type bit = 0; bit<bitstring.size(); bit++)
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

	#ifdef __FAST_BINARY_MATCHING__
	set_match_vector(bitstring);
	#endif
}

void 
fast_ternary_condition::recombine(fast_ternary_condition& offspring, unsigned long method)
{
	assert( (method>=0) && (method<3) );

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
	}
#ifdef __FAST_BINARY_MATCHING__
	set_match_vector(bitstring);
#endif

}

bool 
fast_ternary_condition::is_more_general_than(const fast_ternary_condition& cond) 
const
{
	string		str = cond.string_value();
	unsigned long 	bit;
	unsigned long	sz = bitstring.size();
	bool		does_subsume;

	bit = 0;
	does_subsume = true;
	while  ( (bit<sz) && (does_subsume) )
	{
		does_subsume = ((bitstring[bit]=='#') || (str[bit]==bitstring[bit]));
		bit++;
	}

	return does_subsume;
}

void
fast_ternary_condition::random() 
{
	string::size_type	bit;

	bitstring = "";

	for(bit = 0; bit<fast_ternary_condition::no_bits; bit++)
	{
		if (xcs_random::random()<fast_ternary_condition::dont_care_prob)
		{
			bitstring += '#';
		} else {
			bitstring += '0' + xcs_random::dice(2);
		}
	}

#ifdef __FAST_BINARY_MATCHING__
	set_match_vector(bitstring);
#endif	
}

/// single point crossover
void	
fast_ternary_condition::single_point_crossover(fast_ternary_condition& offspring)
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
fast_ternary_condition::two_point_crossover(fast_ternary_condition& offspring)
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
fast_ternary_condition::uniform_crossover(fast_ternary_condition& offspring)
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
fast_ternary_condition::generality() const
{
	return 1-specificity();
};

//! specificity
double 
fast_ternary_condition::specificity() const
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

#ifdef __FAST_BINARY_MATCHING__
void
fast_ternary_condition::set_match_vector(const string &str)
{
        assert(str.size()==__BIT_CONDITION_SIZE__);

        for(int ch=0; ch<str.size(); ch++)
        {
                long bit = str.size()-1-ch;

                switch (str[ch])
                {
                        case '0': first_part[bit] = 0; second_part[bit] = 0; break;
                        case '1': first_part[bit] = 1; second_part[bit] = 1; break;
                        case '#': first_part[bit] = 1; second_part[bit] = 0; break;
                }
        }
	
// 	check_match_vector();
}

//! check whether bitstring and the match vectors are coherent
void 
fast_ternary_condition::check_match_vector() const
{
        assert(bitstring.size()==__BIT_CONDITION_SIZE__);

	string str = "";

        for(int ch=(__BIT_CONDITION_SIZE__-1); ch>=0; ch--)
        {
		if (first_part[ch]==0 && second_part[ch]==0)
			str += "0"; 
		else if (first_part[ch]==1 && second_part[ch]==1)
			str += "1"; 
		else str += "#";
	}

	assert(str==bitstring);
};
#endif
