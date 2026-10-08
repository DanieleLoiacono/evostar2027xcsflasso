#include <sstream>
#include <iostream>
#include <iterator>
#include <cassert>
#include "xcs_utility.h"
#include "xcs_random.h"
#include "pf/nlms.h"

using namespace std;

namespace xcsflib
{

bool	nlms_pf::init = false;
double	nlms_pf::learning_rate;
double	nlms_pf::xzero;


nlms_pf::nlms_pf( void * owner ) : base_pf( owner )
{
	assert(init);
	for (int i=0; i<=dimension; i++)
	{
		weights.push_back(0);
	}
}

nlms_pf::nlms_pf(xcslib::configuration_manager& xcs_config)
{
	// assert(base_pf::inited());
	if (dimension==0)
	{
		xcs_utility::error(class_name(), "constructor", "section <prediction::based> must be inited before <"+tag_name()+">", 1);			
	}

	if ( !nlms_pf::init )
	{
		if (!xcs_config.exist(tag_name()))
		{
			xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
		}
		
		try {
			learning_rate = xcs_config.Value(tag_name(), "learning rate");
		} catch (...) {
			xcs_utility::error(class_name(), "constructor", "attribute \'learning rate\' not found in <" + tag_name() + ">", 1);
		}

		try {
			xzero = xcs_config.Value(tag_name(), "x0");
		} catch (...) {
			xcs_utility::error(class_name(), "constructor", "attribute \'x0\' not found in <" + tag_name() + ">", 1);
		}

		// not supported
		xcs_utility::set_flag(xcs_config.Value(tag_name(), "random weights", "off"),flag_random_weights);

	}	
	nlms_pf::init = true;
}

void nlms_pf::recombine (base_pf *f)
{
	/*! nothing done during recombination
	 *  we tried both mixing the weights and averaging them but the performance appers not to be influenced
	 */	
}

base_pf*
nlms_pf::clone(void *owner) const
{
	nlms_pf *f = new nlms_pf(owner);

	f->weights.clear();

	for (int i=0; i<dimension+1; i++)
	{
		f->weights.push_back(this->weights[i]);
	}
	return f;
}

void 
nlms_pf::update(const vector <double> &inputs, double target)
{
	vector<double> preprocessed_inputs; 

	if (degree>1)
		polynomial(inputs, preprocessed_inputs);
	else
		preprocessed_inputs = std::move(inputs);

	double error = target - output(inputs);

	double xpow2 = xzero*xzero;
	for (int i=0;i<dimension;i++)
	{
		xpow2 += preprocessed_inputs[i]*preprocessed_inputs[i];
	}

	double correction = (learning_rate*error)/xpow2;

	weights[0] += xzero * correction;
	for (int i=0; i<dimension; i++)
	{
		weights[i+1]+=correction*preprocessed_inputs[i];
	}
}

void 
nlms_pf::print( ostream & output ) const
{
	output <<"[";
	for (int i=0; i<=dimension; i++)
	{
		if (i!=dimension)
			output << weights[i] << ";";
		else
			output << weights[i] << "]";
	}
}

void 
nlms_pf::read( istream & input )
{
	weights.clear();
	weights.reserve(dimension);
	char dummy;

	input >> dummy >> weights[0];

	assert(dummy=='[');

	for (int i=1; i<=dimension; i++)
	{
		if (i!=dimension)
		{
			input >> weights[i];
		} else {
			input >> weights[i] >> dummy;
			assert(dummy==']');
		}
	}
}

//! output the prediction value
double 
nlms_pf::output (const vector <double> &inputs) const
{
	// clog << "INPUT SIZE = " << inputs.size() << endl;

	vector<double> preprocessed_inputs; 

	if (degree>1)
		polynomial(inputs, preprocessed_inputs);
	else
		preprocessed_inputs = std::move(inputs);

	// clog << "DEGREE = " << degree << endl;
	// clog << "INPUT SIZE " << inputs.size() << endl;
	// clog << "PREPROCESSED INPUT SIZE = " << preprocessed_inputs.size() << endl;
	// clog << "DIMENSION = " << dimension << endl;
	// clog << "weights.size() = " << weights.size() << endl;

	assert (preprocessed_inputs.size () == dimension);
	assert (preprocessed_inputs.size () == (weights.size()-1));

	double x = xzero * weights[0];
	for (int i = 0; i < dimension; i++)
	{
		x += preprocessed_inputs[i] * weights[i+1];
	}

	if (isnan(x))
	{
		assert(false);
		cout << "NAN - INPUT ";
 		copy (preprocessed_inputs.begin(), preprocessed_inputs.end(), ostream_iterator<double>(cout," "));
		cout << " - WEIGHTS ";
 		copy (weights.begin(), weights.end(), ostream_iterator<double>(cout," "));
		cout << endl;
	}
	return x;
};
	
//! latex string representing the prediction function
string 
nlms_pf::latex_equation(int precision) const
{
	ostringstream str;
	str.setf( ios::fixed );
	str.precision(precision);
	str << weights[0] * xzero;
	for(int i=1; i<=dimension; i++)
	{
		str << "+";
		str << weights[i];
		str << "\\times";
		str << "x_{" << i << "}";
	} 
	return str.str();
}

//! string representing the prediction function
string 
nlms_pf::equation() const
{
	ostringstream str;
	str << weights[0] << "*" << xzero;
	for(int i=1; i<=dimension; i++)
	{
		str << "+" << "x" << i;
		str << "*" << weights[i];
	} 
	return str.str();
}

//! put the weights to zero
void 
nlms_pf::clear ()
{
	for (int i = 0; i < dimension + 1; i++)
	{
		weights[i] = 0;
	}
}

} //! end xcsflib namespace
