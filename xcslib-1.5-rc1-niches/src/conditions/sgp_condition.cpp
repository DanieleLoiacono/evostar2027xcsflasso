#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include "xcs_random.h"
#include "configuration_manager.h"
#include "xcs_definitions.h"
#include "sgp_condition.h"

using namespace std;

bool				sgp_condition::init = false;
rpn_interpreter*	sgp_condition::interpreter;
unsigned long		sgp_condition::no_variables;
unsigned long		sgp_condition::no_constants;
unsigned long		sgp_condition::condition_size; 
double				sgp_condition::prob_covering;
double				sgp_condition::prob_dontcare;
unsigned long		sgp_condition::no_covering_or;
unsigned long 		sgp_condition::random_tree_size;
bool				sgp_condition::constrained_mutation = false;
double				sgp_condition::variable_covering_probability;

sgp_condition::covering_type sgp_condition::covering_method = covering_type::MINTERMS;
sgp_condition::mutation_type sgp_condition::mutation_method = mutation_type::SUBTREE_CROSSOVER;

sgp_condition::sgp_condition()
{
	if (!sgp_condition::init)
	{
		xcs_utility::error(class_name(),"sgp_condition2()", "not inited", 1);
	}
}

void sgp_condition::set_parameters(xcslib::configuration_manager &xcs_config)
{
	string str_instruction_set;
	string str_mutation_type;
	string str_covering_type;

	try {
		str_instruction_set = (string) xcs_config.Value(tag_name(), "instruction set");
	} catch (...) {
		xcs_utility::error(class_name(), "constructor", "attribute \'instruction set\' not found in <" + tag_name() + ">", 1);
	}

	try {
		sgp_condition::no_variables = (unsigned long) xcs_config.Value(tag_name(), "number of variables");
	} catch (...) {
		xcs_utility::error(class_name(), "constructor", "attribute \'number of variables\' not found in <" + tag_name() + ">", 1);
	}

	try {
			sgp_condition::no_constants = (unsigned long) xcs_config.Value(tag_name(), "number of constants");
	} catch (...) {
		xcs_utility::error(class_name(), "constructor", "attribute \'number of constants\' not found in <" + tag_name() + ">", 1);
	}

	try {
			sgp_condition::condition_size = (unsigned long) xcs_config.Value(tag_name(), "condition size");
	} catch (...) {
		xcs_utility::error(class_name(), "constructor", "attribute \'condition size\' not found in <" + tag_name() + ">", 1);
	}

	try {
			variable_covering_probability = (double) xcs_config.Value(tag_name(), "variable covering probability");
	} catch (...) {
		string str_parameter = "variable covering probability";
		xcs_utility::error(class_name(), "constructor", "attribute \'"+str_parameter+"\' not found in <" + tag_name() + ">", 1);
	}

	if ((variable_covering_probability<=0)||(variable_covering_probability>1))
	{
		xcs_utility::error(class_name(), "constructor", "\'variable covering probability\' must be in (0,1] in <" + tag_name() + ">", 1);		
	}

	sgp_condition::prob_dontcare = 1 - variable_covering_probability;

	str_mutation_type = (string) xcs_config.Value(tag_name(), "mutation","subtree-crossover");

	// try {
	// 		str_mutation_type = (string) xcs_config.Value(tag_name(), "mutation","subtree-crossover");
	// } catch (...) {
	// 	xcs_utility::error(class_name(), "constructor", "attribute \'mutation\' not found in <" + tag_name() + ">", 1);
	// }

	str_covering_type = (string) xcs_config.Value(tag_name(), "covering", "randomized-minterms");

	// try {
	// 		str_covering_type = (string) xcs_config.Value(tag_name(), "covering");
	// } catch (...) {
	// 	xcs_utility::error(class_name(), "constructor", "attribute \'covering\' not found in <" + tag_name() + ">", 1);
	// }

	random_tree_size = (unsigned long) xcs_config.Value(tag_name(), "random tree size", (unsigned long) (0.2*condition_size));

	sgp_condition::no_covering_or = (unsigned long) xcs_config.Value(tag_name(), "number of covering OR", (unsigned long) 0);

	if (str_mutation_type == "subtree-crossover")
	{
		mutation_method = mutation_type::SUBTREE_CROSSOVER;
	} else if (str_mutation_type == "subtree-replacement")
	{
		mutation_method = mutation_type::SUBTREE_REPLACEMENT;
	} else if (str_mutation_type == "constrained-node-replacement") {
		mutation_method = mutation_type::CONSTRAINED_NODE_REPLACEMENT;
	} else if (str_mutation_type == "unconstrained-node-replacement") {
		mutation_method = mutation_type::UNCONSTRAINED_NODE_REPLACEMENT;
	} else {
		cout << "READ <" << str_mutation_type << ">" << endl;
		xcs_utility::error(class_name(), "constructor", "supported mutation: subtree-crossover, subtree-replacement, constrained-node-replacement or unconstrained-node-replacement", 1);
	}

	if (str_covering_type == "minterms")
	{
		covering_method = covering_type::MINTERMS;
	} else if (str_covering_type == "randomized-minterms") {
		covering_method = covering_type::RANDOMIZED_MINTERMS;
	} else if (str_covering_type == "random-clauses") {
		covering_method = covering_type::RANDOM_CLAUSES;
	} else {
		xcs_utility::error(class_name(), "constructor", "supported covering: minterms, randomized-minterms, or random-clauses", 1);
	}

	sgp_condition::prob_covering = 1 - sgp_condition::prob_dontcare;
	sgp_condition::interpreter = new rpn_interpreter(str_instruction_set,no_variables,no_constants);
}

sgp_condition::sgp_condition(xcslib::configuration_manager& xcs_config)
{
	char		out[2048];
	string		str_instruction_set;
	string		str_mutation_type;
	
	char		str_only_and[16];
	ifstream 	config;
	
	if (!sgp_condition::init)
	{
		if (!xcs_config.exist(tag_name()))
		{
			xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
		}
		
		set_parameters(xcs_config);

		init = true;
	}
};

sgp_condition::~sgp_condition()
{

}

bool 
sgp_condition::operator<(const sgp_condition& cond) const
{
	//!
	return string_value()<cond.string_value();
	// return string_condition < cond.string_condition;
};

bool 
sgp_condition::operator==(const sgp_condition& cond) const
{
	//!
	return (condition==cond.condition);
};

bool 
sgp_condition::operator!=(const sgp_condition& cond) const
{
	//!
	return (condition!=cond.condition);
}

bool
sgp_condition::match(const binary_inputs& inputs) const
{
	assert(inputs.allow_numeric_representation());
	vector<long> numerical_inputs;

	inputs.numeric_representation(numerical_inputs);

	return ((sgp_condition::interpreter->execute(condition, numerical_inputs))!=0);
}

void
sgp_condition::cover(const binary_inputs& inputs) 
{
	assert(inputs.allow_numeric_representation());
	vector<long> numerical_inputs;

	inputs.numeric_representation(numerical_inputs);

	switch (covering_method)
	{
		case sgp_condition::covering_type::MINTERMS:
			cover_rpn_condition_minterms(condition,numerical_inputs);
			break;
		case sgp_condition::covering_type::RANDOMIZED_MINTERMS:
			cover_rpn_condition_randomized_minterms(condition,numerical_inputs);
			break;
		case sgp_condition::covering_type::RANDOM_CLAUSES:
			cover_rpn_condition_randomized_minterms(condition,numerical_inputs);
			break;
	}

	update_string_value();

	assert ((sgp_condition::interpreter->execute(condition, numerical_inputs))!=0);
}

void
sgp_condition::mutate(double mutation_rate, const binary_inputs &inputs)
{
	mutate(mutation_rate);
}

void
sgp_condition::mutate(double mutation_rate)
{
	switch (mutation_method)
	{
		case mutation_type::SUBTREE_CROSSOVER:
			mutate_using_crossover(mutation_rate);
			// clog << "CROSSOVER MUTATION" << endl;
			break;
		case mutation_type::SUBTREE_REPLACEMENT:
			mutate_subtree_replacement(mutation_rate);
			// clog << "SUBTREE MUTATION" << endl;
			break;
		case mutation_type::UNCONSTRAINED_NODE_REPLACEMENT:
			// clog << "NOTE UNCONSTRAINED MUTATION" << endl;
			mutate_node_replacement(mutation_rate);
			break;
		case mutation_type::CONSTRAINED_NODE_REPLACEMENT:
			// clog << "NOTE CONSTRAINED MUTATION" << endl;
			mutate_node_replacement(mutation_rate);
			break;
	}

	update_string_value();
}

void sgp_condition::mutate_subtree_replacement(double mutation_rate)
{
	vector<unsigned long> mutated_condition; 
	sgp_condition	random_condition;

	if (xcs_random::random()<mutation_rate)
	{
		long root_position = xcs_random::dice(condition.size());
		long start_position = get_subtree_end(condition,root_position);
		long end_position = root_position;

		//! generate a random condition
		random_condition.random();

		for(unsigned long i=0; i<start_position+1; i++)
			mutated_condition.push_back(condition[i]);
		for(unsigned long i=0; i<random_condition.condition.size(); i++)
			mutated_condition.push_back(random_condition.condition[i]);
		for(unsigned long i=end_position+1; i<condition.size(); i++)
			mutated_condition.push_back(condition[i]);
	}

	condition = mutated_condition;
}

void
sgp_condition::mutate_node_replacement(double mutation_rate = 0.0)
{
	double allele_mutation_probability = mutation_rate;

	if (mutation_rate == 0)
	{
		allele_mutation_probability = 1/double(condition.size());
	}

	for (int i=0; i<condition.size();i++)
	{
		if (xcs_random::random()<allele_mutation_probability)
		{
			if (constrained_mutation)
			{
				condition[i] = interpreter->constrained_random_token(condition[i]);
			}
			else
			{
				condition[i] = interpreter->unconstrained_random_token();
			}				
		}
	}
}

void
sgp_condition::mutate_using_crossover(double mutation_rate)
{
	sgp_condition	random_condition;

	if (xcs_random::random()<mutation_rate)
	{
		//! generate a random condition
		random_condition.random();

		//! recombinate the random condition with this condition
		recombine(random_condition);
	}
}

void 
sgp_condition::recombine(sgp_condition& parent, unsigned long method)
{
	/*
	//! implements a single point crossover
        unsigned long sz = condition.size();
	unsigned long szp = parent.condition.size();

	unsigned long cut = 1+xcs_random::dice(condition.size()-1);
	unsigned long cut_parent = 1+xcs_random::dice(parent.condition.size()-1);

	vector<unsigned long> offspring1;
	vector<unsigned long> offspring2;
	
	//! heads first
	//! copy heads
	for(unsigned long pos=0; pos<cut+1; pos++)
		offspring1.push_back(condition[pos]);
	for(unsigned long pos=0; pos<cut_parent+1; pos++)
		offspring2.push_back(parent.condition[pos]);
	//! copy tails
	for(unsigned long pos=cut_parent+1; pos<szp; pos++)
		offspring1.push_back(parent.condition[pos]);
	for(unsigned long pos=cut+1; pos<sz; pos++)
		offspring2.push_back(condition[pos]);

	condition = offspring1;
	parent.condition = offspring2;
	*/

	//! implements a two point crossover
    unsigned long sz = condition.size();
	unsigned long szp = parent.condition.size();

	unsigned long cut1 = 1+xcs_random::dice(condition.size()-1);
	unsigned long cut2 = 1+xcs_random::dice(condition.size()-1);
	unsigned long cut_parent1 = 1+xcs_random::dice(parent.condition.size()-1);
	unsigned long cut_parent2 = 1+xcs_random::dice(parent.condition.size()-1);

	if (cut1>cut2) swap(cut1,cut2);
	if (cut_parent1>cut_parent2) swap(cut_parent1,cut_parent2);

	vector<unsigned long> offspring1;
	vector<unsigned long> offspring2;
	
	//! heads first
	//! copy heads
	for(unsigned long pos=0; pos<cut1+1; pos++)
		offspring1.push_back(condition[pos]);
	for(unsigned long pos=0; pos<cut_parent1+1; pos++)
		offspring2.push_back(parent.condition[pos]);

	//! copy middle
	for(unsigned long pos=cut_parent1+1; pos<cut_parent2; pos++)
		offspring1.push_back(parent.condition[pos]);
	for(unsigned long pos=cut1+1; pos<cut2; pos++)
		offspring2.push_back(condition[pos]);

	//! copy tails
	for(unsigned long pos=cut2; pos<sz; pos++)
		offspring1.push_back(condition[pos]);
	for(unsigned long pos=cut_parent2; pos<szp; pos++)
		offspring2.push_back(parent.condition[pos]);

	condition = offspring1;
	update_string_value();

	parent.condition = offspring2;
	parent.update_string_value();
}

void
sgp_condition::random() 
{
	//! to check the mincount meaning
	sgp_condition::interpreter->random(condition, random_tree_size, 5);
	update_string_value();	
}

void 
sgp_condition::print(ostream& output) 
const 
{
	output << string_value();
}

//! set the condition to a value represented as a string
void
sgp_condition::set_string_value(const string& string_value)
{
	istringstream input(string_value);
	unsigned long token;
	char column;

	condition.clear();

	if (input>>token)
	{
		condition.push_back(token);
		while (input>>column>>token)
		{
			if (column!=':')
			{
				xcs_utility::error(class_name(),"class constructor", "invalid separator", 1);
			}
			condition.push_back(token);
		}
	}
}

//! return the condition as a string
string
sgp_condition::string_value()
const
{
	return string_condition;
}

//! return the condition as a string
string
sgp_condition::compute_string_value()
const
{
	ostringstream str;

	str << "";

	if (condition.size()==0)
		return str.str();

	str << condition[0];
	unsigned long	var;
	for(var = 1; var<condition.size(); var++)
	{
		str << ':';
		str << condition[var];
	}
	return str.str();
}

void
sgp_condition::cover_rpn_condition_minterms(vector<unsigned long> &program, vector<long> &numerical_inputs, double prob_covering)
{
	//! first implementation: covering is done as in the usual XCS
	
	vector<long>		selected;	//! variables selected for covering

	program.clear();

	add_minterm(program, numerical_inputs, prob_covering);

	// even if the number of or clauses is zero, at least one clause must be added
	for (unsigned long no=1; no<sgp_condition::no_covering_or+1; no++)
	{
		add_minterm(program, numerical_inputs, prob_covering);
		program.push_back(sgp_condition::interpreter->code_function("or"));
	}
}

void
sgp_condition::add_minterm(vector<unsigned long> &program, const vector<long> &numerical_inputs, double prob_covering)
{
	vector<long> selected_inputs;	//! variables selected for covering

	do {
		selected_inputs.clear();
		for(long v=0; v<numerical_inputs.size(); v++)
		{
			if (xcs_random::random()<=prob_covering)
			{
				selected_inputs.push_back(v);
			}
		}
		// cout << "SELECTED " << selected_inputs.size() << endl;
	} while (selected_inputs.size()==0);

	std::shuffle(selected_inputs.begin(), selected_inputs.end(), xcs_random::rng());

	for(size_t i=0; i<selected_inputs.size(); i++)
	{
		program.push_back(sgp_condition::interpreter->code_variable(selected_inputs[i]));

		if (numerical_inputs[selected_inputs[i]]==0)
		{
			program.push_back(sgp_condition::interpreter->code_function("not"));
		} 
	}

	// adds all the AND on the top of the stack
	for(unsigned long v=0; v<selected_inputs.size()-1; v++)
	{
		program.push_back(sgp_condition::interpreter->code_function("and"));
	}
}

void
sgp_condition::cover_rpn_condition_randomized_minterms(vector<unsigned long> &program, vector<long> &numerical_inputs, double prob_covering)
{
	vector<long>		selected;	//! variables selected for covering

	program.clear();

	add_randomized_minterm(program, numerical_inputs, prob_covering);

	// even if the number of or clauses is zero, at least one clause must be added
	for (unsigned long no=1; no<sgp_condition::no_covering_or+1; no++)
	{
		add_randomized_minterm(program, numerical_inputs, prob_covering);
		program.push_back(sgp_condition::interpreter->code_function("or"));
	}
}

void
sgp_condition::add_randomized_minterm(vector<unsigned long> &program, const vector<long> &numerical_inputs, double prob_covering)
{
	vector<long> selected_inputs;

	do {
		selected_inputs.clear();
		for(long v=0; v<numerical_inputs.size(); v++)
		{
			if (xcs_random::random()<=prob_covering)
			{
				selected_inputs.push_back(v);
			}
		}
	} while (selected_inputs.size()==0);

	std::shuffle(selected_inputs.begin(), selected_inputs.end(), xcs_random::rng());

	program.push_back(sgp_condition::interpreter->code_variable(selected_inputs[0]));
	if (numerical_inputs[selected_inputs[0]]==0)
	{
		program.push_back(sgp_condition::interpreter->code_function("not"));
	}
	
	for(size_t i=1; i<selected_inputs.size(); i++)
	{
		program.push_back(sgp_condition::interpreter->code_variable(selected_inputs[i]));

		if (numerical_inputs[selected_inputs[i]]==0)
		{
			program.push_back(sgp_condition::interpreter->code_function("not"));
		} 

		program.push_back(sgp_condition::interpreter->code_function("and"));
	}
}

void
sgp_condition::add_random_clause(vector<unsigned long> &program, const vector<long> &numerical_inputs, double prob_covering)
{
	vector<long> selected_inputs;

	do {
		selected_inputs.clear();
		for(long v=0; v<numerical_inputs.size(); v++)
		{
			if (xcs_random::random()<=prob_covering)
			{
				selected_inputs.push_back(v);
			}
		}
	} while (selected_inputs.size()==0);

	std::shuffle(selected_inputs.begin(), selected_inputs.end(), xcs_random::rng());

	if (selected_inputs.size()==1)
	{
		program.push_back(sgp_condition::interpreter->code_variable(selected_inputs[0]));
		if (numerical_inputs[selected_inputs[0]]==0)
		{
			program.push_back(sgp_condition::interpreter->code_function("not"));
		}
	} else {
		for(size_t i=0; i<selected_inputs.size(); i+=2)
		{
			if (i==(selected_inputs.size()-1))
			{
				program.push_back(sgp_condition::interpreter->code_variable(selected_inputs[0]));
				if (numerical_inputs[selected_inputs[0]]==0)
				{
					program.push_back(sgp_condition::interpreter->code_function("not"));
				}
			} else {
				long input0 = numerical_inputs[selected_inputs[i]];
				long input1 = numerical_inputs[selected_inputs[i+1]];

				switch (xcs_random::dice(4))

				program.push_back(sgp_condition::interpreter->code_variable(selected_inputs[i]));

				if (numerical_inputs[selected_inputs[i]]==0)
				{
					program.push_back(sgp_condition::interpreter->code_function("not"));
				} 

				program.push_back(sgp_condition::interpreter->code_function("and"));
			}
			
		}
	}
	
}

long 
sgp_condition::get_subtree_end(const vector<unsigned long> &program, long index) const
{
	int modifier = -1; // -1 if going backward

	if (interpreter->is_variable(program[index]))
	{
		return index+1*modifier;
	}

	if (interpreter->is_constant(program[index]))
	{
		return index+1*modifier;
	}

	if (interpreter->is_function(program[index]))
	{
		unsigned long start_next_argument = get_subtree_end(program, index+1*modifier);

		if (interpreter->arity(program[index])==1)
			return start_next_argument;
		else
			return get_subtree_end(program, start_next_argument);
	}
}
