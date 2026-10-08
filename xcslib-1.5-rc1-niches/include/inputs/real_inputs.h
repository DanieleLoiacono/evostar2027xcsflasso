#ifndef __REAL_INPUTS__
#define __REAL_INPUTS__

#define __STATE_OPEN_CHAR__ '('
#define __STATE_CLOSE_CHAR__ ')'
#define __STATE_SEPARATOR_CHAR__ ','

#include <iostream>
#include <string>
#include <vector>

#include "inputs_base.h"
#include "xcs_utility.h"

using namespace std;

#define __INPUTS_VERSION__ "vector of reals (class long_sensors)"

class real_inputs : public inputs_base<real_inputs, double>
{

private:
	//! current sensor value
	vector<double> inputs;

	//! number of input values
	unsigned int no_inputs;

public:

	//! constructor
	real_inputs() {no_inputs = 1; inputs.reserve(1);};

	real_inputs(const real_inputs& ls);

	real_inputs(string value) { no_inputs = 0; inputs.clear(); set_string_value(value);};

	real_inputs(unsigned long dim) { no_inputs = dim; inputs.reserve(dim);};

	real_inputs(vector<double> real_values) { no_inputs = real_values.size(); inputs = real_values;};

	//! destructor
	virtual ~real_inputs() { inputs.clear(); }

	//! name of the class that implements the sensors
	string class_name() { return ("real_inputs"); };

	//! return the sensor size; for usual long sensors, returns the number of bits.
	unsigned long size() const {return inputs.size();};

	// //! return the value of the specified input position
	// double input(unsigned long) const;

	// //! set the value of a specific input
	// void set_input(unsigned long, double);

	//! set the value of the sensors from a string
	string string_value() const;

	//! set the value of the sensors from a string
	void set_string_value(const string &str);

	//! set no_inputs of input
	void set_dim(unsigned long dim)
	{
		no_inputs = dim;
                inputs.clear();
                inputs.reserve(dim);
	};

	//! equality operator
	bool operator==(const real_inputs& sens) const;

	//! not equal operator
	bool operator!=(const real_inputs& sens) const;

	//! assignment operators
	real_inputs& operator=(real_inputs& sens);
	real_inputs& operator=(const real_inputs& sens);

	double get_vect_length() const
	{
		double result=0;
		for (int i=0; i< no_inputs; i++)
			result+=inputs[i]*inputs[i];
		return result;
	}

	//! return true if the sensory inputs can be represented as a vector of long
	bool allow_numeric_representation() const { return true; };

	//! return a vector of double that represents the sensory inputs
	vector<double> numeric_representation() const;

	//! set the sensory inputs as a vector of long
	void set_numeric_representation(const vector<double>& value);
};
#endif
