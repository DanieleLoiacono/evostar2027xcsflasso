#ifndef __PF_VALUE__
#define __PF_VALUE__

#include <cassert>
#include <iostream>

#include "base.h"
#include "configuration_manager.h"

namespace xcsflib
{

class value_pf : public base_pf
{

	private:

		//! true when the function has been initialized
		static bool	init;

		//! learning rate eta
		static double	learning_rate;

		//! true if MAM is used for prediction
		static bool	flag_use_mam;

		//! init prediction value
		static double init_value;

		//! prediction value
		double prediction;

	public:
		//! class name
		string class_name () const {return string ("xcsf::constant_pf");};

		//! tag name
		string tag_name () const {return string ("prediction::value");};

		//! true if the class has been initialized
		bool inited() const {return init;};

		//! constructor 
		value_pf(xcslib::configuration_manager &xcs_config);

		//! constructor
		value_pf(void *owner);

		//! clone the current function
		base_pf *clone (void *owner = NULL) const;

		//! output the prediction value
		double output (const vector < double >&input) const;
		
		//! parameter update based on input values and target value
		void update(const vector <double> &input, double target);

		//! recombine function
		virtual void recombine (base_pf *f);
		
		//! mutate function (does nothing)
		base_pf *mutate () {assert(false);};

		//! print function to stream
		void print(ostream &output) const;

		//! read function from stream
		void read(istream &input);
		
		//! clear function parameters
		void clear();
		
		//! return a string representing the prediction function
		string equation() const;
		
		//! return a latex string for prediction function
		string latex_equation(int precision) const;
};

}	//! end namespace
#endif
