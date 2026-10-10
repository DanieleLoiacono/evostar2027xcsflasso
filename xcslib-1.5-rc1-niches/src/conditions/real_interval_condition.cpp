#include <cassert>
#include <string>
#include <fstream>
#include <algorithm>
#include "xcs_definitions.h"
#include "xcs_random.h"
#include "configuration_manager.h"
#include "real_interval_condition.h"


using namespace std;

#define __INTERVAL_TEXT_FORMAT__ "[%lf;%lf]"
#define __INTERVAL_GRAPH_OPEN_CHAR__ '|'
#define __INTERVAL_GRAPH_CLOSE_CHAR__ '|'
#define __INTERVAL_GRAPH_FULL_CHAR__ 'O'
#define __INTERVAL_GRAPH_PART_CHAR__ 'o'
#define __INTERVAL_GRAPH_NOT_CHAR__ '.'

#define __REAL_INTERVAL_CONDITION_SEPARATOR__ ','


bool				real_interval_condition::init = false;
unsigned long		real_interval_condition::no_inputs;
double		        real_interval_condition::min_input;
double  	  		real_interval_condition::max_input;
double  			real_interval_condition::r0;
double  			real_interval_condition::m0;
bool 	 			real_interval_condition::bounded;

real_interval_condition::t_mutation_type 	real_interval_condition::mutation_type;
real_interval_condition::t_crossover_type	real_interval_condition::crossover_type;

real_interval_condition::real_interval_condition()
{
	if (!real_interval_condition::init)
	{
		xcs_utility::error(class_name(),"real_interval_condition()", "not inited", 1);
	}
	value.reserve(no_inputs);
}

real_interval_condition::real_interval_condition(unsigned long no_inputs_, double min_input_, double max_input_, double r0_, double m0_,
	bool bounded_ = true,
real_interval_condition::t_mutation_type mutation_type_ = real_interval_condition::t_mutation_type::fixed, 
real_interval_condition::t_crossover_type crossover_type_ = real_interval_condition::t_crossover_type::one_point) 
{
	init = true;
	no_inputs = no_inputs_;
	min_input = min_input_;
	max_input = max_input_;
	r0 = r0_;
	m0 = m0_;
	bounded = bounded_;
	mutation_type = mutation_type_;
	crossover_type = crossover_type_;
}

real_interval_condition::real_interval_condition(const real_interval_condition& condition)
{
	value = condition.value;
}

void 
real_interval_condition::set_parameters(xcslib::configuration_manager& configuration)
{
	if (!configuration.exist(tag_name()))
	{
		xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
	}
	
	try {
		no_inputs = configuration.Value(tag_name(), "input size");
	} catch (...)
	{
		string msg = "attribute \'input size\' not found in <" + tag_name() + ">";
		xcs_utility::error(class_name(), "constructor", msg, 1);
	}

	try {
		min_input = configuration.Value(tag_name(), "min input");
	} catch (...)
	{
		string msg = "attribute \'min input\' not found in <" + tag_name() + ">";
		xcs_utility::error(class_name(), "constructor", msg, 1);
	}

	try {
		max_input = configuration.Value(tag_name(), "max input");
	} catch (...)
	{
		string msg = "attribute \'max input\' not found in <" + tag_name() + ">";
		xcs_utility::error(class_name(), "constructor", msg, 1);
	}

	if (min_input>=max_input)
	{
		string msg = "\'min input\' must be lower than \'max input\' in <" + tag_name() + ">";
		xcs_utility::error(class_name(), "constructor", msg, 1);
	}

	try {
		r0 = configuration.Value(tag_name(), "r0");
	} catch (...)
	{
		string msg = "attribute \'r0\' not found in <" + tag_name() + ">";
		xcs_utility::error(class_name(), "constructor", msg, 1);
	}

	try {
		m0 = configuration.Value(tag_name(), "m0");
	} catch (...)
	{
		string msg = "attribute \'m0\' not found in <" + tag_name() + ">";
		xcs_utility::error(class_name(), "constructor", msg, 1);
	}

	string is_bounded = configuration.Value(tag_name(), "bounded", "off");
	bounded = xcs_utility::string_to_boolean_option(is_bounded);

	string str_mutation_type = (string) configuration.Value(tag_name(), "mutation", "fixed");
	string str_crossover_type = (string) configuration.Value(tag_name(), "crossover", "one-point");

	set_mutation_type(str_mutation_type);
	set_crossover_type(str_crossover_type);
}

real_interval_condition::real_interval_condition(xcslib::configuration_manager& configuration)
{
	if (!real_interval_condition::init)
	{
		set_parameters(configuration);
	}
	init = true;
}

void real_interval_condition::print_parameters(ostream& OUTPUT) 
const
{
	OUTPUT << "<" << tag_name() << ">" << endl;
	OUTPUT << "input size = " << no_inputs << endl;
	OUTPUT << "min input = " << min_input << endl;
	OUTPUT << "max input = " << max_input << endl;
	OUTPUT << "r0 = " << r0 << endl;
	OUTPUT << "m0 = " << m0 << endl;
	OUTPUT << "bounded = " << (bounded?"on":"off") << endl;

	OUTPUT << "crossover = ";
	switch (crossover_type)
	{
		case t_crossover_type::one_point:
			OUTPUT << "one-point" << endl;
			break;
		case t_crossover_type::two_point:
			OUTPUT << "two-point" << endl;
			break;
		case t_crossover_type::uniform:
			OUTPUT << "uniform" << endl;
			break;
	}

	OUTPUT << "mutation = ";
	switch (mutation_type)
	{
		case t_mutation_type::proportional:
			OUTPUT << "proportional" << endl;
			break;
		case t_mutation_type::fixed:
			OUTPUT << "fixed" << endl;
			break;
		case t_mutation_type::gaussian:
			OUTPUT << "gaussian" << endl;
			break;
	}	
	OUTPUT << "</" << tag_name() << ">" << endl;
}

bool
real_interval_condition::operator<(const real_interval_condition& cond) const
{
	return (string_value()<cond.string_value());
};

string
real_interval_condition::string_value()
const
{
	int sz = size();
	ostringstream out;
	for (int i=0; i< sz; i++)
	{
		if (i!=sz-1)
			out << value[i].string_value() << __REAL_INTERVAL_CONDITION_SEPARATOR__;
		else
			out << value[i].string_value();
	}
	return out.str();
}

void
real_interval_condition::set_string_value(const string& str)
{
	int pos, old_pos;
	int curr_no_inputs=0;
	old_pos=0;
	pos = str.find(__REAL_INTERVAL_CONDITION_SEPARATOR__);
	while (pos != string::npos)
	{
		string sub_str(str,old_pos,(pos-old_pos));

		xcslib::interval<double> interv;
		interv.set_string_value(sub_str);
		assert(curr_no_inputs < no_inputs);
		value[curr_no_inputs++] = interv;
		old_pos = pos;
		pos = str.find(__REAL_INTERVAL_CONDITION_SEPARATOR__,++old_pos);
	}
	
	if (pos==string::npos)
	{
		string sub_str(str,old_pos,(str.size()-old_pos));
		xcslib::interval<double> interv;
		interv.set_string_value(sub_str);
		assert(curr_no_inputs < no_inputs);
		value[curr_no_inputs++] = interv;
	}
	assert (curr_no_inputs == no_inputs);
}

bool
real_interval_condition::operator==(const real_interval_condition& cond)
const
{
	for (int i=0; i <no_inputs; i++)
	{
		if (value[i]!=cond.value[i])
			return false;
	}
	return true;
};

bool
real_interval_condition::operator!=(const real_interval_condition& condition)
const
{
	for (int interval_i=0; interval_i<no_inputs; interval_i++)
	{
		if (value[interval_i]!=condition.value[interval_i])
			return true;
	}
	return false;
}

real_interval_condition&
real_interval_condition::operator=(const real_interval_condition& condition)
{
	value = condition.value;
	// value.clear();
	// value.reserve(no_inputs);
	// for (int i=0; i<no_inputs; i++)
	// {
	// 	xcslib::interval<double> interv(cond.value[i]);
	// 	value[i]=interv;
	// }
	return (*this);
}

void
real_interval_condition::normalize(vector<double>& input, vector<double>& normalized_input)
{
	assert(false);
	// assert (input.size()==no_inputs);

	// normalized_input.clear();

	// for (int i=0; i<no_inputs; i++)
	// {
	// 	double delta = value[i].get_upper_bound() - value[i].get_lower_bound();
	// 	double curr;
	// 	if (delta!=0)
	// 		curr = (input[i] - value[i].get_lower_bound())/delta;
	// 	else
	// 		curr = 1;
	// 	norm_input.push_back(curr);
    //     }
}

bool
real_interval_condition::match(const real_inputs& input) const
{
	bool result = true;

	vector<double> input_values = input.numeric_representation();

	assert(input.size()== size());

	unsigned long interval_i = 0;

	if (bounded)
	{
		for (size_t interval_i=0; interval_i<no_inputs; interval_i++)
		{
			if (!value[interval_i].contains(input_values[interval_i], min_input, max_input))
			{
				return false;
			}
		}
	} else {
		for (size_t interval_i=0; interval_i<no_inputs; interval_i++)
		{
			if (!value[interval_i].contains(input_values[interval_i]))
			{
				return false;
			}
		}
	}

	return true;
}

void
real_interval_condition::cover(const real_inputs& input)
{
	if (input.size()!=no_inputs)
	{
		cout << "INPUT SIZE = " << input.size() << endl;
		cout << "# INPUTS = " << no_inputs << endl;
	}
	assert (input.size()==no_inputs);

	vector<double> input_values = input.numeric_representation();

    value.clear();
	
	for (unsigned long interval_i=0; interval_i<no_inputs; interval_i++)
	{
		double lower = input_values[interval_i] - r0*xcs_random::random();
		double upper = input_values[interval_i] + r0*xcs_random::random();

		if (bounded)
		{
			lower = std::max(lower,min_input);
			upper = std::min(upper,max_input);
		}

		value.push_back(xcslib::interval<double>(lower,upper));
	}
}

void 
real_interval_condition::gaussian_mutation(double mu)
{
	for (int i=0; i<no_inputs; i++)
	{
		double lower = value[i].get_lower_bound();	
		double upper = value[i].get_upper_bound();

		if (xcs_random::random () < mu)
		{
			lower = lower + m0*(xcs_random::nrandom());
		}

		if (xcs_random::random () < mu)
		{
			upper = upper + m0*(xcs_random::nrandom());
		}

		//! the two bounds mutate independently; if they cross, they are swapped as in check()
		value[i] = xcslib::interval<double>(std::min(lower,upper), std::max(lower,upper));
	}
}

void 
real_interval_condition::fixed_mutation(double mu)
{
	for (int i=0; i<no_inputs; i++)
	{
		double lower = value[i].get_lower_bound();	
		double upper = value[i].get_upper_bound();

		if (xcs_random::random () < mu)
		{
			lower = lower + m0*(xcs_random::random()) * xcs_random::sign();
		}

		if (xcs_random::random () < mu)
		{
			upper = upper + m0*(xcs_random::random()) * xcs_random::sign();
		}

		//! the two bounds mutate independently; if they cross, they are swapped as in check()
		value[i] = xcslib::interval<double>(std::min(lower,upper), std::max(lower,upper));
	}
}

void 
real_interval_condition::proportional_mutation(double mu)
{
	for (int i=0; i<no_inputs; i++)
	{
		if (xcs_random::random () < mu)
		{
			
			//! Butz's mutation ...
			//help <- urand() * (x2-x1)
			double x1,x2;
			
			double range = (value[i].get_upper_bound()-value[i].get_lower_bound());
			
			double center =  xcs_random::random()*range+value[i].get_lower_bound();
				
			//x1 <- help - .5 * (.5+.5*urand()) * (x2-x1)
			//x2 <- help + help - x1 //x1 is the new value of x1
			//-> help is the new middle where urand() is unif. rand. number between 0 and 1
			//now the new x1 and x2:
			//with prob .5 interval size is decreased:
			
			if (xcs_random::random()<.5)
			{	//! decrease
				double sigma = (.5+.5*xcs_random::random()) * range;
				x1 = center - .5 * sigma;
				x2 = center+center-x1;
				//center + .5 * sigma; //x1 is the new value of x1
			} else {//! increase
				double sigma = (1+.5*xcs_random::random()) * range;
				x1 = center - .5 * sigma;
				x2 = center+center-x1;
				//x2 = center + .5 * sigma; //x1 is the new value of x1	
			}
			
			if (bounded)
			{
				x1 = std::max(x1,min_input);
				x2 = std::min(x2,max_input);
			}

			value[i].set_lower_bound(x1);
			value[i].set_upper_bound(x2);
			check(value[i]);
		}
	}
}

void
real_interval_condition::mutate(double mutation_probability, const real_inputs &inputs)
{
	assert (inputs.size() == no_inputs);

	mutate(mutation_probability);
}

void
real_interval_condition::mutate(double mutation_probability)
{
	switch (mutation_type)
	{
		case t_mutation_type::fixed:
			fixed_mutation(mutation_probability);
			break;
		case t_mutation_type::proportional:
			proportional_mutation(mutation_probability);
			break;
		default:
			xcs_utility::error(class_name(),"mutate","unrecognized mutation type",1);
	}
}

void
real_interval_condition::recombine(real_interval_condition& offspring)
{
	switch (crossover_type)
	{
		case t_crossover_type::uniform:
			uniform_crossover(offspring);
			break;
		case t_crossover_type::one_point:
			single_point_crossover(offspring);
			break;
		case t_crossover_type::two_point:
			two_points_crossover(offspring);
			break;
		default:
			xcs_utility::error(class_name(),"mutate","unrecognized mutation type",1);
	}
}

bool
real_interval_condition::is_more_general_than(const real_interval_condition& condition)
const
{
	assert (condition.size() == no_inputs);
	for(unsigned long interval_i=0; interval_i<no_inputs; interval_i++)
	{
		if (!(condition.value[interval_i]<value[interval_i]))
			return false;
	}
	return true;
}

void
real_interval_condition::random()
{
    value.clear();
	double range = max_input - min_input;

	for (int i=0; i<no_inputs; i++)
	{
		double lower_bound = range * xcs_random::random();		
		double upper_bound = lower_bound + (range-lower_bound)*xcs_random::random();
		value.push_back(xcslib::interval<double>(lower_bound, upper_bound));
	}
}

//! uniform crossover
void
real_interval_condition::uniform_crossover(real_interval_condition& offspring)
{
 	unsigned long sz = size();
	
 	for(unsigned long it=0; it<sz; it++)
 	{
 		if (xcs_random::random()<.5)
		{
			//! swap lower
			value[it].swap_lower(offspring.value[it]);
		}
		
		if (xcs_random::random()<.5)
		{
			//! swap upper
			value[it].swap_upper(offspring.value[it]);
		}
		
		check(value[it]);
		check(offspring.value[it]);	
 	}
}


//! single point crossover
void
real_interval_condition::single_point_crossover(real_interval_condition& offspring)
{
	int sz = size();
	int pos;

	pos = xcs_random::dice(sz);

	if (xcs_random::random()<.5)
	{
		value[pos].swap(offspring.value[pos]);
	}
	else
	{
		value[pos].swap_upper(offspring.value[pos]);
		check(value[pos]);
		check(offspring.value[pos]);
	}
		
	for (int i=pos+1; i<no_inputs; i++)
	{
		value[i].swap(offspring.value[i]);
	}
}

//! double point crossover
void
real_interval_condition::two_points_crossover(real_interval_condition& offspring)
{
	int sz = size();
	
	int pos1 = xcs_random::dice(sz);
	int pos2 = xcs_random::dice(sz);

	if (pos1>pos2)
	{
		int tmp;
		tmp = pos1;
		pos1 = pos2;
		pos2 = tmp;
	}


	if (xcs_random::random()<.5)
	{
		value[pos1].swap(offspring.value[pos1]);
	}
	else
	{
		value[pos1].swap_upper(offspring.value[pos1]);
		check(value[pos1]);
		check(offspring.value[pos1]);
	}

	for (int i=pos1+1; i<pos2; i++)
	{
		value[i].swap(offspring.value[i]);
	}

	if (xcs_random::random()<.5)
	{
		value[pos2].swap(offspring.value[pos2]);
	}
	else
	{
		value[pos2].swap_lower(offspring.value[pos2]);
		check(value[pos2]);
		check(offspring.value[pos2]);
	}
}

void 
real_interval_condition::check(xcslib::interval<double>& it)
{
	//! sort the lower and upper boundaries
	double lb = it.get_lower_bound();
	double ub = it.get_upper_bound();
	
	//! if bound are unsorted
	if (lb>ub)
	{
		it.set_lower_bound(ub);	
		it.set_upper_bound(lb);	
	}
	
	//! intervals are limited to [min_input, max_input] only when conditions are bounded
	if (!bounded) return;

	if (it.get_lower_bound()<real_interval_condition::min_input)
	{
		it.set_lower_bound(real_interval_condition::min_input);
	}
	
	if (it.get_upper_bound()>real_interval_condition::max_input)
	{
		it.set_upper_bound(real_interval_condition::max_input);
	}
}

void 
real_interval_condition::set_crossover_type(string str_crossover)
const
{
	if (str_crossover=="one-point")
	{
		crossover_type = crossover_type = t_crossover_type::one_point;
	} else if (str_crossover=="two-point") 
	{
		crossover_type = crossover_type = t_crossover_type::two_point;
	} else if (str_crossover=="uniform") 
	{
		crossover_type = crossover_type = t_crossover_type::uniform;
	} else {
		xcs_utility::error(class_name(), "constructor", "crossover method \'"+str_crossover+"\' not supported in <" + tag_name() + ">", 1);
	}
};

void 
real_interval_condition::set_mutation_type(string str_mutation) const
{
	if (str_mutation=="fixed")
	{
		mutation_type = t_mutation_type::fixed;

	} else if (str_mutation=="proportional")
	{
		mutation_type = t_mutation_type::proportional;

	} else if (str_mutation=="gaussian")
	{
		mutation_type = t_mutation_type::gaussian;

	} else {
		xcs_utility::error(class_name(), "constructor", "mutation method \'"+str_mutation+"\' not supported in <" + tag_name() + ">", 1);
	}
};
