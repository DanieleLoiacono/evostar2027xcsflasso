#include <iostream>
#include <string>
#include "inputs_base.h"
#include "xcs_utility.h"

#ifndef __BINARY_INPUTS__
#define __BINARY_INPUTS__

//! help string for the class
const string __INPUTS_VERSION__ = "binary {0,1} string (class binary_inputs)";

using namespace std;
//using namespace xcs;

class binary_inputs : public inputs_base<binary_inputs, char>
{
private:
	//! current state value
	string	value;

public:
	//! constructor
	binary_inputs() {};

	binary_inputs(const string& value) { set_string_value(value);};

	//! destructor
	virtual ~binary_inputs() {};

	//! name of the class that represents the state
	string class_name() { return ("binary_inputs"); };

	//! return the size of the state; for instance, with binary representation, returns the number of bits.
	unsigned long	size() const {return value.size();};

	//! set the value of the state from a string
	string string_value() const { return value; };

	//! set the value of the state from a string
	void set_string_value(const string &str);

	// //! return the value of the specified input bit
	// char input(unsigned long) const;

	// //! set the value of a specific input
	// void set_input(unsigned long, char);

	//! equality operator 
	bool operator==(const binary_inputs& st) const {
		return (string_value()==st.string_value());
	};

	//! not equal operator
	bool operator!=(const binary_inputs& st) const {
		return (string_value()!=st.string_value());
	};

	//! assignment operators
	binary_inputs& operator=(binary_inputs& st) { value = st.string_value(); return (*this); };
	binary_inputs& operator=(const binary_inputs& st) { value = st.string_value(); return (*this); };

	//! return true if the sensory inputs can be represented as a vector of long
	bool allow_numeric_representation() const { return true; };

	//! return a vector of long that represents the sensory inputs
	void numeric_representation(vector<long>&) const;
	void numeric_representation(vector<double>&) const;

	//! return a vector of long that represents the sensory inputs
	void set_numeric_representation(const vector<long>&);
	void set_numeric_representation(const vector<double>&);
};
#endif
