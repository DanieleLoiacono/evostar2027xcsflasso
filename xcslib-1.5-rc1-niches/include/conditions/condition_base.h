/*!
 * \class condition_base condition_base.h
 *
 * \brief template class for classifier conditions
 */

#ifndef __XCS_CONDITION__
#define __XCS_CONDITION__

#include <string>
#include "xcs_definitions.h"

template <class _condition, class _inputs>
class condition_base
{
public:
	/*!
	 * \fn string class_name() const
	 * \brief name of the class that implements the environment
	 * 
	 * This function returns the name of the class. 
	 * Every class must implement this method that is used to 
	 * trace errors.
	 */
	//! name of the class that implements the condition
	virtual string class_name() const = 0;
	
	//! tag used to access the configuration file
	virtual string tag_name() const = 0;

	//! return the condition as a string
	virtual string string_value() const = 0;

	//! set the condition to a value represented as a string
	virtual void set_string_value(const string& string_value) = 0;

	//! return the condition size
	virtual unsigned long size() const = 0;

	//! true if the representation allow the use of GA subsumption.
	virtual bool allow_ga_subsumption() const = 0; //{return false;};
	
	//! less than operator
	virtual bool operator< (const _condition&) const = 0;

	//! equality operator
	virtual bool operator==(const _condition&) const = 0;

	//! inequality operator
	virtual bool operator!=(const _condition&) const = 0;

	//! assignment operator for a constant value
	virtual _condition& operator=(const _condition&) = 0;

	//! returns true if the condition matches the input configuration
	virtual bool match(const _inputs& inputs) const = 0;

	//! set the condition to cover the input 
	virtual void cover(const _inputs& inputs) = 0;

	//! mutate the condition according to the mutation rate \emph mu
	virtual void mutate(double mutation_rate) = 0;

	//! mutate the condition according to the mutation rate \emph mu; the mutating bits
	virtual void mutate(double mutation_rate, const _inputs& inputs) = 0;

	//! recombine the condition (usually calls the one with the specified method)
	virtual void recombine(_condition& condition) = 0;

	// //! recombine the condition according to the strategy specified with the method variable
	// virtual void recombine(_condition& condition, unsigned long method=0) = 0;
	
	//! pretty print the condition to the output stream "output".
	virtual void print(ostream& output) const = 0;

	//! write the condition to an output stream 
	friend ostream& operator<<(ostream& output, const _condition& condition)
	{
		output << condition.string_value();
		return (output);
	};

	//! read the condition from an input stream 
	friend istream& operator>>(istream& input, _condition& condition)
	{
		string	str;
		if (input >> str)
		{
			condition.set_string_value(str);
		} else {
			xcs_utility::error(condition.class_name(), "operator>>", "condition reading from streaming failed",1);
		}

		return (input);
	};

	//! return true if the condition is subsumed by \emph cond
	virtual bool subsumed_by(const _condition& condition) const {return false;};

	//! return true if the condition is more general than \emph cond
	// virtual bool is_more_general_than(const _condition& condition) const {return false;};

	//! generate a random condition
	virtual void random() = 0;

	//! virtual destructor
	virtual ~condition_base() {};	
};
#endif
