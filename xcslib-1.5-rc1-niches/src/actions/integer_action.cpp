//-------------------------------------------------------------------------
// Filename      : integer_action.cpp
//
// Purpose       : implementation of the integer actions class
//                 
// Special Notes : 
//                 
//
// Creator       : Pier Luca Lanzi
//
// Creation Date : 2002/06/10
//
// Current Owner : Pier Luca Lanzi
//
//-------------------------------------------------------------------------

/*!
 * \file integer_action.cpp
 *
 * \brief implements integer actions used for instance in woods environments
 *
 */

#include <sstream>
#include <string>
#include "xcs_random.h"
#include "integer_action.h"

unsigned long	integer_action::no_actions;

bool integer_action::init = false;

integer_action::integer_action()
{
	if (!integer_action::init)
	{
		xcs_utility::error(class_name(),"integer_action()", "not inited", 1);
	} else {

	}
}

integer_action::integer_action(int act)
{
	if (!integer_action::init)
	{
		xcs_utility::error(class_name(),"integer_action(int)", "not inited", 1);
	} else {
		action = act;
		integer_action::init=true;
	}
}

integer_action::integer_action(xcslib::configuration_manager& xcs_config)
{
	string		input_configuration;
	
	if (!integer_action::init)
	{

		//! look for the init section in the configuration file
		if (!xcs_config.exist(tag_name()))
		{
			xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
		}
	
		try {
			no_actions = xcs_config.Value(tag_name(), "number of actions");
			integer_action::init=true;
		} catch (const char *attribute) {
			string msg = "attribute \'" + string(attribute) + "\' not found in <" + tag_name() + ">";
			xcs_utility::error(class_name(), "constructor", msg, 1);
		}
	} else {
		xcs_utility::error(class_name(),"integer_action(xcs_config_mgr)", "already inited", 1);
	}
}

void 
integer_action::print_parameters(ostream& OUTPUT) 
const
{
	OUTPUT << "<" << tag_name() << ">" << endl;
	OUTPUT << "number of actions = " << no_actions << endl;
	OUTPUT << "</" << tag_name() << ">" << endl;
};

void
integer_action::mutate(const double& mu)
{		
	if (xcs_random::random()<mu)
	{
		action=xcs_random::dice(integer_action::no_actions);
	}
}

string 
integer_action::string_value() const
{	
	ostringstream sstr;
	string	  str;
	sstr << action;

	str = sstr.str();
	
	return str;
}

void 
integer_action::set_string_value(string str)
{
	action = atoi(str.c_str()) % actions();
}
