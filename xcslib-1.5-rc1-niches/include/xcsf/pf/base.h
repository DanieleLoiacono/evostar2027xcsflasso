#ifndef __BASE_AF__
#define __BASE_AF__

#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>

#include "configuration_manager.h"

namespace xcsflib
{
	enum class prediction_function_type
	{
		PREDICTION_NLMS,	//! normalized least mean square
		PREDICTION_RLS,		//! recursive least squares
		PREDICTION_RLSK, 	//! recursive least squares with forgetting factor and kalman
		PREDICTION_VALUE,	//! constant activation function
	};

	static const map<string,prediction_function_type> prediction_function =
	{ 
		{ "value", prediction_function_type::PREDICTION_VALUE},
		{ "nlms", prediction_function_type::PREDICTION_NLMS},
		{ "rls", prediction_function_type::PREDICTION_RLS},
		{ "rlsk", prediction_function_type::PREDICTION_RLSK}
	};

	enum class prediction_function_init_type
	{
		PREDICTION_RANDOM_INIT,	
		PREDICTION_DEFAULT_INIT
	};

	static const map<string,prediction_function_init_type> prediction_function_init =
	{ 
		{ "random", prediction_function_init_type::PREDICTION_RANDOM_INIT},
		{ "default", prediction_function_init_type::PREDICTION_DEFAULT_INIT},
	};

	enum class prediction_function_mutation_type
	{
		PREDICTION_NO_MUTATION, PREDICTION_RANDOM_MUTATION
	};

	static const map<string,prediction_function_mutation_type> prediction_function_mutation =
	{ 
		{ "random", prediction_function_mutation_type::PREDICTION_RANDOM_MUTATION},
		{ "none", prediction_function_mutation_type::PREDICTION_NO_MUTATION},
	};

	class base_pf
	{

	private:
		//! true if the class has been inited
		static bool init;

	protected:

		//! number of input variables 
		static unsigned long degree;

		//! number of input variables 
		static unsigned long dimension;

		//! mutation type 
		static prediction_function_mutation_type mutation_type;

		//! initialization type 
		static prediction_function_init_type init_mode;

		//! default prediction function
		static prediction_function_type default_function;

	public:
		//! pointer to owner classifier
		void *owner;

		//! constructor
		base_pf ()
		{
			assert (init);	//! check if already inited
			owner = NULL;
		}

		//! destructor
		virtual ~base_pf(){};

		//! constructor based on the configuration file
		base_pf (xcslib::configuration_manager& xcs_config);

		//! constructor
		base_pf (void *owner)
		{
			assert (init);		//! check if already inited
			this->owner = owner;	//! set owner
		};

		//! copy constructor
		base_pf (const base_pf & a)
		{
			assert (init);		//! check if already inited
			this->owner = a.owner;	//! set owner
		};

		//! class name
		virtual string class_name () const { return string ("xcsflib::base_pf"); };

		//! class tag
		virtual string tag_name () const { return string ("prediction::base"); };

		//! return an instance 
		friend base_pf *get_prediction_function (void *owner);

		//! return a copy of the current function
		virtual base_pf *clone (void *owner = NULL) const { assert (false); };

		//! set the pointer to the owner classifier
		void set_owner (void *owner) { this->owner = owner; };

		//! return the pointer to the owner classifier
		void *get_owner () { return owner; };

		//! compute classifier prediction
		virtual double output (const vector < double >&input) const { assert (false); };

		//! update prediction function parameters according to a target and gradient
		virtual void update (const vector<double> &input, double t, double g)	{ update(input, t); };
		virtual void update (const vector<double> &input, double t)	{ assert (false); };

		//! pretty print the prediction function parameters to an output stream
		virtual void print (ostream & output) const { assert (false); };

		//! read the prediction function parameters from input stream
		virtual void read (istream & input) { assert (false); };

		//! mutate prediction function 
		virtual base_pf *mutate () { assert (false); };

		//! recombine prediction functions
		virtual void recombine (base_pf * f) { assert(false); };

		//! clear the parameters
		virtual void clear () { assert (false);	};

		//! return the equation corresponding to the prediction function
		virtual string equation () const { assert (false); };
		
		//! true if the class has been initialized
		virtual bool inited () const { return base_pf::init; };

		//! number of inputs involved
		unsigned long dim() const {return dimension;};

		//! preprocess the inputs for the degree
		void polynomial(const vector<double>& inputs, vector<double>& preprocessed_inputs) const;

	};

// base_pf *get_prediction_function_by_type (xcsflib::prediction_function_type function_type, void *owner);

} //! end of xcsflib namespace

#endif
