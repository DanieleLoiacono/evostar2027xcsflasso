#ifndef __BOOLEAN_ACTION__
#define __BOOLEAN_ACTION__

#define __ACTION_VERSION__ "Boolean value 0/1 (class boolean_action)"

#include "action_base.h"
#include "configuration_manager.h"

/*!
 * \class boolean_action
 *
 * \brief definition of Boolean actions
 * \sa action_base
 *
 */

class boolean_action : public virtual action_base<boolean_action>
{
public:
	/*!
	 * \fn string class_name() const
	 * \brief name of the class that implements the action set
	 * 
	 * This function returns the name of the class. 
	 * Every class must implement this method that is used to 
	 * trace errors.
	 */
	string class_name() const { return string("boolean_action"); };

	//! tag used to access the configuration file
	string tag_name() const { return string("boolean_action"); };

	//! default constructor sets the action to 0 (i.e., false)
	boolean_action();

	//! creates an action with the specified value
	boolean_action(int);

	//! init the action parameters from file. In this case, nothing is done.
	boolean_action(xcslib::configuration_manager&);
	
	void print_parameters(ostream& OUTPUT) const { /* nothing to print */};

	//! mutate the action according to the mutation rate
	void mutate(const double&);

	//! specify that Boolean actions can have two values only (0 and 1).
	unsigned long actions() const {return 2;};

	//! return the action value as a string
	string string_value() const;

	//! set the action value from a string
	void set_string_value(string str);
};
#endif
