#include "action_base.h"
#include "xcs_utility.h"
#include "configuration_manager.h"

#ifndef __BINARY_ACTION__
#define __BINARY_ACTION__

#define __ACTION_VERSION__ "binary string (class binary_action)"

using namespace std;

class binary_action : public virtual action_base<binary_action>
{
		
private:
	static bool				init;
	static unsigned long	no_actions;
	static unsigned long	no_bits;
	string					bitstring;
	const static std::vector<std::string> supported_configuration_parameters;

public:
	string class_name() const { return string("binary_action"); };
	string tag_name() const { return string("action::binary"); };

	binary_action();
	binary_action(int);
	binary_action(xcslib::configuration_manager&);

	void print_parameters(ostream& OUTPUT) const;

	~binary_action();
	

	unsigned long actions() const;

	void set_value(unsigned long);
	string string_value() const;
	void set_string_value(string);

	void random();
	void mutate(const double&); 
};

#endif
