/*!
 * \class integer_action integer_action.h
 *
 * \brief implements integer actions used for instance in woods environments
 * \sa action_base
 *
 */

#define __ACTION_VERSION__ "integer (class integer_action)"

#ifndef __INTEGER_ACTION__
#define __INTEGER_ACTION__

#include "action_base.h"
#include "rl_definitions.h"
// #include "xcs_utility.h"
#include "configuration_manager.h"

class integer_action : public virtual action_base<integer_action>
{

private:
	static bool init;			//!< true if the class has been already inited
	static unsigned long no_actions;	//!< number of available actions

public:
	/*!
	 * \fn string class_name() const
	 * \brief name of the class that implements the environment
	 *
	 * This function returns the name of the class.
	 * Every class must implement this method that is used to
	 * trace errors.
	 */
	//! name of the class that implements the environment
	string class_name() const { return string("integer_action"); };

	//! tag used to access the configuration file
	string tag_name() const { return string("action::integer"); };

	//! constructor that reads the class parameters through the configuration manager
	/*!
	 *  This is the first constructor that must be used. Otherwise an error is returned.
	 */
	integer_action(xcslib::configuration_manager&);

	void print_parameters(ostream&) const;

	//! constructor that sets the class parameters inline
	integer_action(int act);

	//! default constructor that can be used only after the class has been already initialized through the configuration manager
	integer_action();

	//! return the number of available actions
	unsigned long actions() const {return no_actions;};

	//! mutate the action according to the mutation rate \emph mu
	void mutate(const double&);

	//! return the action value as a string
	string string_value() const;

	//! set the action value from a string
	void set_string_value(string);

	//! assignment operator
	virtual integer_action& operator=(integer_action& action) {set_string_value(action.string_value()); return *this;};

	//! assignment operator for a constant value
	virtual integer_action& operator=(const integer_action& action) {set_string_value(action.string_value()); return *this;};

};
#endif
