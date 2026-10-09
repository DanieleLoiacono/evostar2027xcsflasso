#include <sstream>
#include <iostream>
#include <algorithm>
#include <iterator>

#include "pf/base.h"
#include "pf/utility.h"
#include "xcs_utility.h"
#include "xcs_random.h"

using namespace std;

namespace xcsflib
{
	//! config the available prediction functions
	void init_prediction_functions(xcslib::configuration_manager& xcs_config)
	{
		//! init base class and all the ones available from the configuration file
		base_pf dummy_base(xcs_config);

		//! init all the prediction functions
		// init_prediction_functions(xcs_config);		

		#ifdef __PF_RLS__
		rls_pf	dummy_rls(xcs_config);
		if (dummy_rls.inited())
		{
			available_functions.insert(prediction_function_type::PREDICTION_RLS);
		}
		#endif

		#ifdef __PF_NLMS__
		nlms_pf	dummy_nlms(xcs_config);
		if (dummy_nlms.inited())
		{
			available_functions.insert(prediction_function_type::PREDICTION_NLMS);
		}
		#endif

		#ifdef __PF_VALUE__
		value_pf	dummy_value(xcs_config);
		if (dummy_value.inited())
		{
			available_functions.insert(prediction_function_type::PREDICTION_VALUE);
		}
		#endif

		#ifdef __PF_RLS_DELTA__
		rls_delta_pf	dummy_rls_delta(xcs_config);
		if (dummy_rls_delta.inited())
		{
			available_functions.insert(prediction_function_type::PREDICTION_RLS_DELTA);
		}
		#endif

		#ifdef __PF_RLSK__
		rlsk_pf	dummy_rlsk(xcs_config);
		if (dummy_rlsk.inited())
		{
			available_functions.insert(prediction_function_type::PREDICTION_RLS);
		}
		#endif

		if (available_functions.size()==0)
		{
			xcs_utility::error( "activation", "get_function_type()", "no prediction functions defined", 1 );
		}		
	};

	base_pf* get_prediction_function( void  * owner )
	{
		if ( base_pf::init_mode == prediction_function_init_type::PREDICTION_DEFAULT_INIT)
			return get_prediction_function_by_type( base_pf::default_function, owner );

		xcs_utility::error("activation", "get_an_implementation()", "only default init mode is supported", 1);
	}

	base_pf* get_prediction_function_by_type(prediction_function_type function_type, void *owner)
	{

		if (available_functions.find(function_type)==available_functions.end())
		{
			// ostringstream msg;
			// msg << "function " << function_type << " not available";

			xcs_utility::error("utility", "get_prediction_function_by_type", "function not available", 1);
		}
		
		switch (function_type)
		{
#ifndef XCSF_NLMS_ONLY
			case prediction_function_type::PREDICTION_VALUE:
				return new value_pf(owner);
			case prediction_function_type::PREDICTION_RLS:
				return new rls_pf(owner);
			case prediction_function_type::PREDICTION_RLS_DELTA:
				return new rls_delta_pf(owner);
#endif
			case prediction_function_type::PREDICTION_NLMS:
				return new nlms_pf(owner);

			// optional functions that we rarely use and might be not included in the source code
			#ifdef __PF_RLSK__
				case prediction_function_type::PREDICTION_RLSK:
				return new rlsk_pf(owner);
			#endif

			default:
				xcs_utility::error("utility", "type_to_function()", "type not allowed", 1);
		}
	}
}
