#ifndef __PF_RLS_DELTA__
#define __PF_RLS_DELTA__

/*!
 * Recursive least squares as described in
 *   P.L. Lanzi, D. Loiacono, S.W. Wilson, D.E. Goldberg, "Generalization in the XCSF
 *   classifier system: analysis, improvement, and extension", IlliGAL Report 2005012,
 *   Section 7.3, Eq. 22-25 and Algorithm 5 (XCSFrls).
 *
 *   V0 = delta * I                                   (Eq. 25)
 *   e  = y - w^T phi                                 (a-priori error, Alg. 5 line 2)
 *   beta_rls = 1 + phi^T V phi                       (Alg. 5 line 7)
 *   V  <- V - beta_rls^{-1} V phi phi^T V            (Alg. 5 line 8)
 *   K  = V phi                                       (Alg. 5 line 9, posterior V)
 *   w  <- w + K e                                    (Alg. 5 line 11)
 *
 * with phi = [x0, x1, ..., xn] (x0 is the constant input). Unlike rls_pf (prediction::rls),
 * delta is read from the configuration and no matrix is added to V after the update.
 * No forgetting factor, no process noise. Offspring copy the parent's weights and start
 * from V = delta * I (same policy as rls_pf/rlsk_pf). Weights start at zero.
 *
 * Configuration:
 *   <prediction::rls_delta>
 *     x0 = 1
 *     delta = 1000
 *   </prediction::rls_delta>
 * and 'prediction function = rls_delta' in <prediction::base>.
 */

#include <cassert>
#include <iostream>
#include <vector>

#include "base.h"
#include "configuration_manager.h"

namespace xcsflib
{

class rls_delta_pf : public base_pf
{
	private:
		//! true when the function has been initialized
		static bool	init;

		//! constant input x0
		static double	xzero;

		//! initial scale of V (V0 = delta * I)
		static double	delta;

		//! weight vector [w0, w1, ..., wn]
		vector<double>	weights;

		//! estimate of (X^T X)^{-1}, (dimension+1) x (dimension+1), row-major
		vector<double>	V;

		//! V <- delta * I
		void reset_matrix();

	public:
		string class_name () const {return string ("xcsf::rls_delta_pf");};
		string tag_name () const {return string ("prediction::rls_delta");};
		bool inited() const {return init;};

		//! constructor based on the configuration file (reads x0 and delta)
		rls_delta_pf(xcslib::configuration_manager &xcs_config);

		//! constructor
		rls_delta_pf(void *owner);

		~rls_delta_pf() {}

		base_pf *clone (void *owner = NULL) const;
		double output (const vector < double >&input) const;
		void update(const vector <double> &input, double target);
		virtual void recombine (base_pf *f);
		base_pf *mutate () {assert(false); return NULL;};
		void print(ostream &output) const;
		void read(istream &input);
		void clear();
		string equation() const;
		string latex_equation(int precision) const;

		//! read-only access (tests)
		const vector<double>& get_weights() const {return weights;};
		const vector<double>& get_matrix() const {return V;};
};

}	//! end namespace
#endif
