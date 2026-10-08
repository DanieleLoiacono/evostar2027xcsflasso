#include <sstream>
#include "pf/base.h"
#include "pf/utility.h"
#include "xcs_utility.h"
#include "xcs_random.h"
//#include "xcs_classifier.h"

using namespace std;

namespace xcsflib
{

//! init static variables
bool								base_pf::init = false;		//! true when base function has been inited
unsigned long						base_pf::dimension;			//! number of inputs
unsigned long						base_pf::degree = 1;		//! polynomial degree
prediction_function_mutation_type	base_pf::mutation_type;		//! mutation type (not used)
prediction_function_init_type 		base_pf::init_mode;			//! init procedure for parameter vector
prediction_function_type			base_pf::default_function;	//! default type of prediction function

base_pf::base_pf(xcslib::configuration_manager& xcs_config)
{
	string	str_mutation;			//! mutation type
	string	str_function;			//! default function
	string	str_init_mode;			//! string to set init mode

	if (!xcs_config.exist(tag_name()))
	{
		xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
	}

	try {
		unsigned long original_dimension = xcs_config.Value(tag_name(), "input size");

		base_pf::degree = xcs_config.Value(tag_name(), "degree", (unsigned long) 1);

		base_pf::dimension = original_dimension * degree;

		str_function = (string) xcs_config.Value(tag_name(), "prediction function");

		str_init_mode = (string) xcs_config.Value(tag_name(), "init mode", "default");
		
	} catch (const char *attribute) {
		string msg = "attribute \'" + string(attribute) + "\' not found in <" + tag_name() + ">";
		xcs_utility::error(class_name(), "constructor", msg, 1);
	}

	clog << "ORIGINAL DIMENSION = " << base_pf::dimension << endl;
	
	// we do not implement mutation over the prediction functions
	mutation_type = prediction_function_mutation_type::PREDICTION_NO_MUTATION;

	//! get the default prediction function type
	// default_function = xcsflib::prediction_function[str_function];
	if (prediction_function.count(str_function))
	{
		default_function = xcsflib::prediction_function.at(str_function);
	} else {
		string msg = "prediction function \'" + str_function + "\' not supported in <" + tag_name() + ">";
		xcs_utility::error(class_name(), "constructor", msg, 1);
	}
	
	//! how parameter vectors are initialized
	// init_mode = xcsflib::prediction_function_init[str_init_mode];
	if (prediction_function_init.count(str_init_mode))
	{
		init_mode = xcsflib::prediction_function_init.at(str_init_mode);
	} else {
		string msg = "prediction function init mode \'" + str_init_mode + "\' not supported in <" + tag_name() + ">";
		xcs_utility::error(class_name(), "constructor", msg, 1);
	}
	
	init = true;
}

void 
base_pf::polynomial(const vector<double>& inputs, vector<double>& preprocessed_inputs) const
{
	preprocessed_inputs = std::move(inputs);
	
	if (degree==1)
		return;

	for(int d=2; d<=degree; d++)
	{
		for(int i=0; i<inputs.size(); i++)
		{
			preprocessed_inputs.push_back(pow(inputs[i],double(d)));
		}
	}

	assert(preprocessed_inputs.size()==inputs.size()*degree);	
}





} //! end of xcsflib namespace
