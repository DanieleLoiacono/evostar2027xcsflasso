/*!
 * \class inputs_base.h
 * \brief Base class to define sensory inputs, i.e., states.
 */

#ifndef __XCS_INPUTS_BASE__
#define __XCS_INPUTS_BASE__

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "xcs_utility.h"

using namespace std;

template <class _inputs, class _input>
class inputs_base
{
public:
	//! name of the class that implements the environment
	virtual string class_name() = 0;
	
	//! tag used to access the configuration file
	virtual string tag_name() { return ""; };

	//! return the size of the input state; for instance, for usual binary sensors, it returns the number of bits.
	virtual unsigned long	size() const = 0;

	// //! return the value of a specific input
	// virtual _input input(unsigned long) const = 0;

	// //! set the value of a specific input
	// virtual void set_input(unsigned long, _input) = 0;

	//! set the value of the sensors from a string
	virtual string string_value() const = 0;

	//! set the value of the sensors from a string
	virtual void set_string_value(const string &str) = 0;

	//! equality operator 
	virtual bool operator==(const _inputs& st) const = 0;

	//! not equal operator
	virtual bool operator!=(const _inputs& st) const = 0;
	
	//! assignment operators
	virtual _inputs& operator=(_inputs&) = 0;
	virtual _inputs& operator=(const _inputs&) = 0;

	//! write the sensors to an output stream 
	friend ostream& operator<<(ostream& output, const _inputs& inputs) {
		output << inputs.string_value();
		return (output);
	};

	//! read the sensors from an input stream 
	friend istream& operator>>(istream& input, _inputs& inputs) {
		string	str;
		if (input >> str)
		{
			inputs.set_string_value(str);
		} else {
			xcs_utility::error(inputs.class_name(), "operator>>", "input reading from streaming failed",1);
		}

		return (input);
	};

	//! return true if the sensory inputs can be represented as a vector of long
	virtual bool allow_numeric_representation() const { return false; };

	//! return a vector of values that represents the sensory inputs
	// virtual void numeric_representation(vector<double>& vals) const {assert(false);};
	virtual vector<double> numeric_representation() const {assert(false);};
	
	//! assign the values of the state values from a numerical vector
	virtual void set_numeric_representation(const vector<double>& vals) {assert(false);};
};
#endif
