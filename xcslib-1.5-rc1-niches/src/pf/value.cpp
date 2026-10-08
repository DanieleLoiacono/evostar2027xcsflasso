#include <sstream>
#include <iostream>
#include <iterator>
#include <cassert>
#include "xcs_utility.h"
#include "xcs_random.h"
#include "pf/value.h"

using namespace std;

namespace xcsflib
{

bool	value_pf::init = false;
double	value_pf::learning_rate;
double	value_pf::init_value;
bool	value_pf::flag_use_mam = false;


value_pf::value_pf( void * owner ) : base_pf( owner )
{
	assert(init);
}

value_pf::value_pf(xcslib::configuration_manager& xcs_config)
{
	// assert(base_pf::inited());
	if (dimension==0)
	{
		xcs_utility::error(class_name(), "constructor", "section <prediction::based> must be inited before <"+tag_name()+">", 1);			
	}

	if ( !value_pf::init )
	{
		bool completed = true;

		if (!xcs_config.exist(tag_name()))
		{
			// xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
			value_pf::init = false;
			return;
		}
		
		string str_use_mam = xcs_config.Value(tag_name(), "use MAM", "on");
		xcs_utility::set_flag(str_use_mam, flag_use_mam);

		init_value = xcs_config.Value(tag_name(), "init value", double(0));

		try {
			learning_rate = xcs_config.Value(tag_name(), "learning rate");
		} catch (...) {
			completed = false;
		}

		value_pf::init = true;
	}	
}

void value_pf::recombine (base_pf *f)
{
	/*! nothing done during recombination
	 *  we tried both mixing the weights and averaging them but the performance appers not to be influenced
	 */	
}

base_pf*
value_pf::clone(void *owner) const
{
	value_pf *f = new value_pf(owner);

	f->prediction = this->prediction;

	return f;
}

void 
value_pf::update(const vector <double> &inputs, double target)
{
	this->prediction += learning_rate * (target - this->prediction);

	// if (!flag_use_mam || (((xcsf_classifier*)owner)->experience > ( 1 / learning_rate))
	// {
	// 	this->prediction += learning_rate * ( target - this->prediction );
	// } else {
	// 	this->prediction += ( target - this->prediction ) / ((xcsf_classifier*)owner)->experience;
	// }
}

void 
value_pf::print( ostream & output ) const
{
	output.setf( ios::scientific );
	output << prediction;
}

void 
value_pf::read( istream & input )
{
	input >> prediction;
}

//! output the prediction value
double 
value_pf::output (const vector <double> &inputs) const
{
	// clog << "INPUT SIZE = " << inputs.size() << endl;

	vector<double> preprocessed_inputs; 

	if (degree>1)
		polynomial(inputs, preprocessed_inputs);
	else
		preprocessed_inputs = std::move(inputs);

	// clog << "PREPROCESSED INPUT SIZE = " << preprocessed_inputs.size() << endl;

	assert (preprocessed_inputs.size () == dimension);
	return this->prediction;
};
	
//! latex string representing the prediction function
string 
value_pf::latex_equation(int precision) const
{
	ostringstream str;
	str.setf(ios::fixed);
	str.precision(precision);
	str << this->prediction;
	return str.str();
}

//! string representing the prediction function
string 
value_pf::equation() const
{
	ostringstream str;
	str.setf( ios::fixed );
	str << this->prediction;
	return str.str();}

//! put the weights to zero
void 
value_pf::clear ()
{
	// nothing to do!
}

} //! end xcsflib namespace
