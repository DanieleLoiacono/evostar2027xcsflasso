#include "xcs_random.h"
#include "boolean_action.h"

boolean_action::boolean_action()
{
	action = 0;
};

boolean_action::boolean_action(int act)
{
	if ((act!=1)&&(act!=0))
	{
		xcs_utility::error(class_name(),"class constructor", "value not allowed", 1);
	}
	action = act;
};

boolean_action::boolean_action(xcslib::configuration_manager& xcs_config)
{
	action = 0;
};

void
boolean_action::mutate(const double& mu)
{		
	if (xcs_random::random()<mu)
	{
		action=1-action;
	}
}

string 
boolean_action::string_value() const
{
	if (action) 
		return string("1"); 
	else 
		return string("0");
}

void 
boolean_action::set_string_value(string str)
{
	if (str=="0")
		action = 0;
	else if (str=="1")
		action = 1;
	else {
		xcs_utility::error(class_name(),"set_string_value", "value '"+str+"' not allowed", 1);
	}
}
