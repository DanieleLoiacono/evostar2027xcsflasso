#include <string>
#include <iostream>
#include <cmath>
#include "binary_action.h"
#include "xcs_random.h"
#include "xcs_utility.h"

using namespace std;

// const char* __XCS_BITSTRING_ACTION_CFG_IN__ =	"\n number of bits = %u ";
// const char* __XCS_BITSTRING_ACTION_CFG_OUT__ =	"\t\tnumber of bits = %u\n";
// #define     __XCS_BITSTRING_VARS_IN__ binary_action::no_bits

bool binary_action::init = false;
unsigned long binary_action::no_actions;
unsigned long binary_action::no_bits;
const std::vector<std::string> binary_action::supported_configuration_parameters = {"number of bits"};

binary_action::binary_action()
{
	if (!init)
	{
		xcs_utility::error(class_name(),"binary_action()", "not inited", 1);
	} else {

	}
}

binary_action::binary_action(int act)
{
	if (!init)
	{
		xcs_utility::error(class_name(),"binary_action()", "not inited", 1);
	} else {
		action = act;
		bitstring = xcs_utility::long2binary(act, binary_action::no_bits);
	}
}


unsigned long
binary_action::actions() const
{
	return binary_action::no_actions;
};


binary_action::binary_action(xcslib::configuration_manager& xcs_config)
{
	//! already initialized
	if (init) return;

	if (!xcs_config.exist(tag_name()))
	{
		xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
	}

	xcs_config.check_parameters(tag_name(),supported_configuration_parameters);	

	try {		
		no_bits = xcs_config.Value(tag_name(), "number of bits");
	} catch (const char *attribute) {
		string msg = "attribute \'" + string(attribute) + "\' not found in <" + tag_name() + ">";
		xcs_utility::error(class_name(), "constructor", msg, 1);
	}

	init = true;
	
	no_actions = (unsigned long) pow(double(2),int(binary_action::no_bits));

#ifdef __DEBUG__
		cout << "BITS " << binary_action::no_bits << endl;
		cout << "ACTIONS " << actions() << endl;
#endif
}

void binary_action::print_parameters(ostream& OUTPUT) 
const
{
	OUTPUT << "<" << tag_name() << ">" << endl;
	OUTPUT << "number of bits = " << no_bits << endl;
	OUTPUT << "</" << tag_name() << ">" << endl;
}

binary_action::~binary_action()
{

}

void
binary_action::random()
{		
	string::size_type	bit;

	bitstring = "";

	for(bit=0; bit<binary_action::no_bits; bit++)
	{
		if (xcs_random::random()<.5)
			bitstring += "1";
		else
			bitstring += "0";
	}
	
	action = xcs_utility::binary2long(bitstring);
}

void
binary_action::mutate(const double& mutationRate)
{		
	string::size_type	bit;

	for(bit=0; bit<bitstring.size(); bit++)
	{
		if (xcs_random::random()<mutationRate)
		{
			if (bitstring[bit]=='0')
				bitstring[bit] = '1';
			else
				bitstring[bit] = '0';
		}
	}
	action = xcs_utility::binary2long(bitstring);
}

string 
binary_action::string_value() const
{
	return bitstring;
}

void 
binary_action::set_string_value(string str)
{
	bitstring = str;
	action = xcs_utility::binary2long(str);
}
/*
binary_action&
binary_action::operator=(binary_action& act)
{
	action = act.action; 
	xcs_utility::long2binary(act,no_bits);
	return *this;
}; 

binary_action&
binary_actiono::operator=(const binary_action& act)
{
	action = act.action; 
	xcs_utility::long2binary(act,no_bits);
	return *this;
}*/
void 
binary_action::set_value(unsigned long act) { 
	action = act;
	bitstring = xcs_utility::long2binary(act, binary_action::no_bits);
};
