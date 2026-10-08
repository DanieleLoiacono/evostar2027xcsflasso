#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include "xcs_utility.h"
#include "xcs_random.h"
#include "rpn_interpreter.h"

using namespace std;

//! global functions
//! prototypes
long eval_next_arg(stack<long>&);

void _and(stack<long> &s)
{ 
	long arg1;
	long arg2;

	if (s.size()>1)
	{
		arg1 = eval_next_arg(s);
		arg2 = eval_next_arg(s);
		s.push(min(arg1,arg2));
	}
};


void _or(stack<long> &s)
{ 
	long arg1;
	long arg2;

	if (s.size()>1)
	{
		arg1 = eval_next_arg(s);
		arg2 = eval_next_arg(s);
		s.push(max(arg1,arg2));
	}
};

void _not(stack<long> &s)
{
	long arg;
	if (s.size()>0)
	{
		arg = eval_next_arg(s);
		if (arg!=0)
		{
			s.push(0);
		} else {
			s.push(1);
		}
	}
};

void _add(stack<long> &s)
{ 
	long arg1;
	long arg2;

	if (s.size()>1)
	{
		arg1 = eval_next_arg(s);
		arg2 = eval_next_arg(s);
		s.push(arg1+arg2);
	}
};

void _neg(stack<long> &s)
{
	long arg = eval_next_arg(s);
	s.push(-arg);
}

void _gt(stack<long> &s)
{ 
	long arg1;
	long arg2;

	if (s.size()>1)
	{
		arg1 = eval_next_arg(s);
		arg2 = eval_next_arg(s);
		if (arg1>arg2)
		{
			s.push(1);
		} else {
			s.push(0);
		}
	}
};

void _eq(stack<long> &s)
{ 
	long arg1;
	long arg2;

	if (s.size()>1)
	{
		arg1 = eval_next_arg(s);
		arg2 = eval_next_arg(s);
		if (arg1==arg2)
		{
			s.push(1);
		} else {
			s.push(0);
		}
	}
};

void _eval(stack<long> &s) {};


long 
eval_next_arg(stack<long> &s)
{
	long result;
	result = s.top();
	s.pop();
	return result;
}

//! definition of the instruction set
const rpn_interpreter::t_token		rpn_interpreter::instruction_set[] = \
	{ {"and",2,_and}, {"or",2,_or}, {"not",1,_not}, {"add",2,_add}, {"neg",1,_neg}, {"gt",2,_gt}, {"eq",2,_eq} };
const unsigned long			rpn_interpreter::no_available_functions = sizeof(rpn_interpreter::instruction_set)/sizeof(t_token);

//! class constructor
rpn_interpreter::rpn_interpreter(const string &selection, unsigned long vars, unsigned long cons)
{
	unsigned long f; 

	no_functions = 0;
	no_variables = vars;
	no_constants = cons;
	prob_functions = 0.333;
	prob_variables = 0.333;
	prob_constants = 0.334;

	selected_functions.clear();
	code.clear();

	if (string(selection)=="all")
	{
		//! all the available functions are selected
		for(unsigned long f=0; f<rpn_interpreter::no_available_functions; f++)
		{
			selected_functions.push_back(f);
			code[instruction_set[f].name] = f;
			cout << "\tINSTRUCTION\t" << instruction_set[f].name << endl;
		}
		no_functions = rpn_interpreter::no_available_functions;
	} else {
		//! look for the function name in the list of available functions
		string::size_type cur_pos = 0;		//! current position
		string::size_type prev_pos = 0;		//! previous position
		string name;

		while ( (cur_pos = selection.find_first_of(':', cur_pos))!=string::npos)
		{
			name = selection.substr(prev_pos,cur_pos-prev_pos);
			prev_pos = ++cur_pos;
			cout << "\tINSTRUCTION\t" << name << endl;

			for(f=0; ((f<rpn_interpreter::no_available_functions) && (instruction_set[f].name!=name)); f++);

			if (f==rpn_interpreter::no_available_functions)
			{
				xcs_utility::error(class_name(),"class constructor", "selected instruction <"+name + "> not available of bound", 1);
			}
			selected_functions.push_back(f);
			code[name] = selected_functions.size()-1;
			no_functions++;
		}

		// last function
		name = selection.substr(prev_pos,selection.size());
		for(f=0; ((f<rpn_interpreter::no_available_functions) && (instruction_set[f].name!=name)); f++);

		if (f==rpn_interpreter::no_available_functions)
		{
			xcs_utility::error(class_name(),"class constructor", "selected instruction <"+name + "> not available of bound", 1);
		}
		selected_functions.push_back(f);
		cout << "\tINSTRUCTION\t" << name << endl;

		// clog << "SELECTED FUNCTION <" << name << ">" << endl;
		code[name] = selected_functions.size()-1;
		no_functions++;

	}
	cout << "\tNO FUNCTIONS\t" << no_functions << endl;
	cout << "\tNO VARIABLES\t" << no_variables << endl;
	cout << "\tNO CONSTANTS\t" << no_constants << endl;
	//assert(no_functions<10);
};

int 
rpn_interpreter::arity(long function_index) const
{
	assert(is_function(function_index));

	// cout << instruction_set[selected_functions[function_index]].name << " " << instruction_set[selected_functions[function_index]].arity << endl;
	return instruction_set[selected_functions[function_index]].arity;
}

string
rpn_interpreter::string_value(const vector<unsigned long> &program)
const 
{
	ostringstream result;
	vector<unsigned long>::const_iterator	tk;

	for(tk=program.begin(); tk!=program.end(); tk++)
	{
		if (*tk<no_functions)
		{
			result << instruction_set[selected_functions[*tk]].name << ';';
		} else if (no_variables && (*tk<(no_variables+no_functions))) {
			result << 'X' << *tk-no_functions << ';';
		} else {
			result << *tk-(no_functions+no_variables) << ';';
		}
	}
	return result.str();
};

unsigned long
rpn_interpreter::random_token()
const
{
	double rnd = xcs_random::random();
	if (rnd<prob_functions)
		return xcs_random::dice(no_functions);
	else if (rnd<(prob_functions+prob_variables))
		return (no_functions+xcs_random::dice(no_variables));
	else 
		return (no_functions+no_variables+xcs_random::dice(no_constants));
}

bool 
rpn_interpreter::is_function(unsigned long token) const
{
	return token<no_functions;
}

bool
rpn_interpreter::is_variable(unsigned long token) const
{
	return token>=no_functions && token<(no_functions+no_variables);
}

bool
rpn_interpreter::is_constant(unsigned long token) const
{
	return (token>=(no_functions+no_variables)) && (token<(no_functions+no_variables+no_constants));
}

unsigned long
rpn_interpreter::random_terminal()
const
{
	double rnd = xcs_random::random();
	if (rnd<prob_variables/(prob_variables+prob_constants))
		return (no_functions+xcs_random::dice(no_variables));
	else 
		return (no_functions+no_variables+xcs_random::dice(no_constants));
}

unsigned long
rpn_interpreter::code_function(string name)
{
	return code[name];
}

unsigned long 
rpn_interpreter::code_variable(unsigned long var)
{
	return (no_functions+var);
}

unsigned long
rpn_interpreter::code_constant(unsigned long constant)
{
	return (no_functions+no_variables+constant);
}

long
rpn_interpreter::execute(const vector<unsigned long> &program, const vector<long> &variables)
const 
{
	stack<long> execution_stack;			//! stack used for computation
	vector<unsigned long>::const_iterator	tk;	//! iterator to scan the program
	long result;					//! returned value

	for(tk=program.begin(); tk!=program.end(); tk++)
	{
		if (*tk<no_functions)
		{
			//! execute the function
			(instruction_set[selected_functions[*tk]].function)(execution_stack);
		} else if (no_variables && (*tk<(no_variables+no_functions))) {
			//! if a variable is found, its value is pushed onto the stack
			execution_stack.push(variables[*tk-no_functions]);
		} else {
			//! constants are pushed onto the stack
			execution_stack.push(*tk-no_functions-no_variables);
		}
	}
	
	if (execution_stack.size()==0)
	{
		return 0;
	} else {
		return execution_stack.top();
	}
};

void
rpn_interpreter::random(vector<unsigned long> &program, unsigned long size, unsigned long mincount)
const
{
	unsigned long		current_size = 0;
	long				stack_count = 0;

	vector<unsigned long>::iterator	st;

	program.clear();

	while (1)
	{
		current_size++;
		if ( ((current_size + labs(stack_count))>=size) ||
		     (labs(stack_count)==mincount) )
		{
			program.push_back(random_terminal());
			stack_count += 1;

		} else if (stack_count==0) {
			unsigned long  fun = random_function();

			program.push_back(fun);
			
			stack_count += 1 - instruction_set[selected_functions[fun]].arity;

		} else {
			unsigned long token = random_token();

			program.push_back(token);
			
			if (token<no_functions)
			{
				stack_count += 1 - instruction_set[selected_functions[token]].arity;
			} else {
				stack_count++;
			}
		}

		if (current_size>=size)
			break;
	}

	reverse(program.begin(),program.end());

	clog << "RANDOM PROGRAM GENERATED OF SIZE " << program.size() << endl;
	
	assert(program.size()<=size);
};
void
rpn_interpreter::set_probability(const double pf, const double pv)
{
	prob_functions = pf;
	prob_variables = pv;
	prob_constants = 1-(pf+pv);
};
