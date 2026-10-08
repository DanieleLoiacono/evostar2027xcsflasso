#include "action_base.h"
#include "xcs_utility.h"
#include "configuration_manager.h"

#ifndef __DUMMY_ACTION__
#define __DUMMY_ACTION__



// const string __ACTION_VERSION__ = "dummy action (class dummy action)";

class dummy_action : public virtual action_base<dummy_action>
{

public:
	string class_name() const { return string("dummy_action"); };	
	string tag_name() const { return string("dummy_action"); };

	dummy_action();

	dummy_action(xcslib::configuration_manager&) {};
	
	dummy_action(int){};

	~dummy_action(){};

	unsigned long actions() const {return 1;};

	//! return the integer action value
	unsigned long value() const { return 0; };

	void set_value(unsigned long){};

	string string_value() const {return "#";};

	void set_string_value(string) {};

	void random(){};

	void mutate(const double& mu){};

	//! nothing to print for a dummy action
	void print_parameters(ostream& OUTPUT) const {};

	//! equal operator
	bool operator==(const dummy_action& act) const { return true; };

	//! less than operator
	virtual bool operator< (const dummy_action& act) const { return false; };

	//! not equal operator
	virtual bool operator!=(const dummy_action& act) const { return false; };

};

#endif
