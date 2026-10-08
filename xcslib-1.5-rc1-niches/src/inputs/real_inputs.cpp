#include <string>
#include <sstream>
#include <cassert>
#include <vector>

#include "real_inputs.h"

real_inputs::real_inputs(const real_inputs& rs)
{
        this->no_inputs = rs.no_inputs;
        this->inputs = rs.inputs;
}

void
real_inputs::set_string_value(const string &str)
{	
	inputs.clear();
	istringstream in(str);
	double val;
	long dim=0;

	while (in>>val)
	{
		inputs.push_back(val);
		dim++;
	}
	no_inputs = dim;
}

string
real_inputs::string_value() const
{
	ostringstream out;
	out << inputs[0];
	for (int i=1; i<no_inputs; i++)
	{
		out << ' ' << inputs[i];
	}
	return out.str();
}

// double
// real_inputs::input(unsigned long pos)
// const
// {
//         assert(pos<no_inputs);
//         return inputs[pos];
// }

// void
// real_inputs::set_input(unsigned long pos, double val)
// {
//         assert(pos<no_inputs);
//         inputs[pos] = val;
// }

real_inputs&
real_inputs::operator=(real_inputs& real_inputs_)
{
	this->no_inputs = real_inputs_.no_inputs;
	this->inputs = real_inputs_.inputs;
	// inputs.clear();
	// for (int i=0; i<no_inputs; i++)
	// 	this->inputs.push_back(sens.inputs[i]);
	return (*this);
}

real_inputs&
real_inputs::operator=(const real_inputs& real_inputs_)
{
	this->no_inputs = real_inputs_.no_inputs;
	this->inputs = real_inputs_.inputs;

	// inputs.clear();
	// for (int i=0; i<no_inputs; i++)
	// 	this->inputs.push_back(sens.inputs[i]);
	return (*this);
}

bool
real_inputs::operator==(const real_inputs& sens)
const
{
        assert(no_inputs == sens.no_inputs);
        for (int i=0; i<no_inputs; i++)
        {
                if (inputs[i]!=sens.inputs[i])
                        return false;
        }
        return true;
}

bool
real_inputs::operator!=(const real_inputs& sens)
const
{
        assert(no_inputs == sens.no_inputs);
        for (int i=0; i<no_inputs; i++)
        {
                if (inputs[i]!=sens.inputs[i])
                        return true;
        }
        return false;
}

// void
// real_inputs::numeric_representation(vector<double>& value)
// const
// {
//     value = inputs;
// }

vector<double> 
real_inputs::numeric_representation()
const
{
	return inputs;
}

void
real_inputs::set_numeric_representation(const vector<double>& value)
{		
        inputs = value;

		//! this should be eliminated it does not make sense to cache this value
		no_inputs = inputs.size();
		// clog << "set_numeric_representation: value size = " << value.size() << endl;
		// clog << "set_numeric_representation: inputs size = " << inputs.size() << endl;
}
