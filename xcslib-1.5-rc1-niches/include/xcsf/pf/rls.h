#ifndef __PF_RLS__
#define __PF_RLS__

#include <cassert>
#include <iostream>

extern "C" {
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
}

#include "base.h"
#include "configuration_manager.h"

namespace xcsflib
{

class rls_pf : public base_pf
{

	private:

		//! true when the function has been initialized
		static bool	init;

		//! x0
		static double	xzero;

		//! weight vector
		vector<double>	weights;

		//! delta
		static double	delta;

		//! true if weights are randomly initialized
		bool 		flag_random_weights;

		//! matrix for RLS implementation
		gsl_matrix *V; 

	public:
		//! class name
		string class_name () const {return string ("xcsf::rls_pf");};

		//! tag name
		string tag_name () const {return string ("prediction::rls");};

		//! true if the class has been initialized
		bool inited() const {return init;};

		//! constructor 
		rls_pf(xcslib::configuration_manager &xcs_config);

		//! constructor
		rls_pf(void *owner);

		//! destructor
		~rls_pf()
		{
			if (init)
			{
				if (V)
					gsl_matrix_free(V);
			}
		}

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
