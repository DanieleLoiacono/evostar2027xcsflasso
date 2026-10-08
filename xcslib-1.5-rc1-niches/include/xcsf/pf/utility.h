#ifndef	__PF_UTILITY__
#define	__PF_UTILITY__

#include <cassert>
#include <cmath>
#include <iostream>
#include <set>
using namespace std;

#include "prediction_functions.h"
#include "configuration_manager.h"

namespace xcsflib
{
	//! set of available functions whose configuration is specified in the confsys file
	static set<prediction_function_type>	available_functions;

	//! look for the function configurations in the confsys file
	void init_prediction_functions(xcslib::configuration_manager& xcs_config);

	//! get a prediction function based on the initialization procedure (default or random)
	base_pf* get_prediction_function (void *owner);

	//! get a prediction function of a specific type
	base_pf* get_prediction_function_by_type(prediction_function_type function_type, void* owner);
}
#endif
