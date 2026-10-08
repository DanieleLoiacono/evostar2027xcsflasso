#include <sstream>
#include <iostream>
#include <iterator>
#include <cassert>
#include "xcs_utility.h"
#include "xcs_random.h"
#include "xcsf_classifier.h"
#include "pf/rlsk.h"

extern "C" {
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
}

using namespace std;

namespace xcsflib
{

bool	rlsk_pf::init = false;
double	rlsk_pf::xzero;
double	rlsk_pf::delta;
double	rlsk_pf::Q;
double	rlsk_pf::lambda;
double	rlsk_pf::ilambda;
bool	rlsk_pf::flag_use_covariance = false;
bool	rlsk_pf::flag_use_forgetting = false;
bool	rlsk_pf::flag_r = false;



rlsk_pf::rlsk_pf( void * owner ) : base_pf( owner )
{
	assert(init);
	for (int i=0; i<=dimension; i++)
	{
		weights.push_back(0);
	}
	V = gsl_matrix_alloc(dimension+1, dimension+1);
	gsl_matrix_set_identity(V);
	gsl_matrix_scale(V, (double) delta);
}

rlsk_pf::rlsk_pf(xcslib::configuration_manager& xcs_config)
{
	if (dimension==0)
	{
		xcs_utility::error(class_name(), "constructor", "section <prediction::based> must be inited before <"+tag_name()+">", 1);			
	}

	string str_kalman; 

	if ( !rlsk_pf::init )
	{
		if (!xcs_config.exist(tag_name()))
		{
			// xcs_utility::warning(class_name(), "constructor", "section <" + tag_name() + "> not found");
			rlsk_pf::init = false;
			return;
		}
		
		try {
			delta = xcs_config.Value(tag_name(), "delta");
		} catch (...) {
			xcs_utility::error(class_name(), "constructor", "attribute \'delta\' not found in <" + tag_name() + ">", 1);			
		}

		try {
			Q = xcs_config.Value(tag_name(), "Q");
		} catch (...) {
			xcs_utility::error(class_name(), "constructor", "attribute \'Q\' not found in <" + tag_name() + ">", 1);			
		}

		try {
			lambda = xcs_config.Value(tag_name(), "lambda");
		} catch (...) {
			xcs_utility::error(class_name(), "constructor", "attribute \'lambda\' not found in <" + tag_name() + ">", 1);			
		}

		try {
			str_kalman = (string) xcs_config.Value(tag_name(), "kalman");
		} catch (...) {
			xcs_utility::error(class_name(), "constructor", "attribute \'kalman\' not found in <" + tag_name() + ">", 1);			
		}

		xcs_utility::set_flag(str_kalman,flag_r);

		if (lambda>0)
		{
			flag_use_forgetting = true;
			ilambda = 1/lambda;		//! 1/lambda
		}

		if (Q>0)
		{
			flag_use_covariance = true;
		}

		clog << "*** forgetting = " << string(flag_use_forgetting?"on":"off") << endl;
		clog << "*** covariance = " << string(flag_use_covariance?"on":"off") << endl;

		V = gsl_matrix_alloc(dimension+1, dimension+1);
		gsl_matrix_set_identity(V);
		gsl_matrix_scale(V, (double) delta);

		rlsk_pf::init = true;
	}	
}

void rlsk_pf::recombine (base_pf *f)
{
	/*! nothing done during recombination
	 *  we tried both mixing the weights and averaging them but the performance appers not to be influenced
	 */	
}

base_pf*
rlsk_pf::clone(void *owner) const
{
	rlsk_pf *f = new rlsk_pf(owner);

	f->weights.clear();
	for (int i=0; i<dimension+1; i++)
	{
		f->weights.push_back(this->weights[i]);
	}
	return f;
}

void 
rlsk_pf::update(const vector <double> &input, double target)
{
	double error = target;

	//! builds input vector (phi) and compute the error
	gsl_vector *phi = gsl_vector_alloc(dimension+1);
	gsl_vector_set(phi,0,xzero);
	error -= xzero*weights[0];    

	for (int i=0; i<dimension; i++)
	{
		gsl_vector_set(phi,i+1,input[i]);	//! phi(i) = x(i)
		error -= input[i]*weights[i+1];		//! e_t
	}

	//! numerator
	gsl_vector *numerator = gsl_vector_alloc(dimension+1);
	gsl_blas_dgemv(CblasNoTrans, 1.0, V, phi, 0.0, numerator); // temp_vect = V*phi

	//! denominator = 1+...
	double denominator=0;
	gsl_blas_ddot(phi, numerator, &denominator); // beta = phi^T * temp_vect = phi^T * V * phi

	double _R;

	if (!flag_r)
	{
		_R = 1.;
	} else {
		_R = ((xcsf_classifier*)this->owner)->qerror;
		_R = max(_R, 0.01*0.01);	//! set a lower limit for the contribution

		//cerr << "CONTRIBUTION " << _R << endl;
	}

	if (!flag_use_forgetting)
	{
		denominator += _R; // beta = 1 + beta = 1 + phi^T * V * phi
	} else {

		//!  1 + lambda^{-1}*phi^T * V * phi
		denominator = _R + ilambda*denominator;

		// cerr << "*** LAMBDA " << ilambda << " DEN ORIGINALE " << denominator << endl;
	}
    
	//! compute K(t)
	gsl_vector *K = gsl_vector_alloc(dimension+1);

	gsl_vector_memcpy (K, numerator);	//! K <- numerator

	if (flag_use_forgetting)
	{
		gsl_vector_scale(K, ilambda);
		//assert(false);
	}

	gsl_vector_scale (K, 1/denominator);	//! K <- K/denominator


	//! update V(t)

	//! product K(t)x^T which produces a matrix
	gsl_matrix *KxT = gsl_matrix_alloc(dimension+1,dimension+1);
	gsl_matrix_set_zero (KxT);

	gsl_blas_dger(1.0, K, phi, KxT);	//! KxT <- KxT + K * x^T

	//! then K(t) x^T V
	gsl_matrix *KxTV = gsl_matrix_alloc(dimension+1,dimension+1);
	gsl_matrix_set_zero (KxTV);

	//! KxTV <- KxT * V 
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., KxT, V, 1., KxTV); 

	//! V <- V - K(t) x^T V
	gsl_matrix_sub (V, KxTV);  

	if (flag_use_forgetting)
	{
		gsl_matrix_scale(V,ilambda);
	}

	if (flag_use_covariance)
	{
		//! add Q
		gsl_matrix *identity = gsl_matrix_alloc(dimension+1,dimension+1);
		gsl_matrix_set_identity(identity);
		gsl_matrix_scale (V, Q);		//! multiply I*Q in case we use a fixed variance
		gsl_matrix_add (V, identity);  
		gsl_matrix_free(identity);	//! I matrix
	}

	
	//! w(t) <- K(t)*e(t)    
	for (int i=0; i<=dimension; i++)
	{
		weights[i] += gsl_vector_get(K,i)*error;
	}


	//! free 
	gsl_vector_free(phi);		//! free the input phi(t)
	gsl_vector_free(numerator);	//! free the numerator for  K(t)
	gsl_vector_free(K);		//! free the gain  K(t)

	gsl_matrix_free(KxT);		//! temp matrix KxT
	gsl_matrix_free(KxTV);		//! temp matrix KxTV
	
}

void 
rlsk_pf::print( ostream & output ) const
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
rlsk_pf::read( istream & input )
{
	weights.clear();
	weights.reserve(dimension);

	char dummy;
	input >> dummy >> weights[0];

	for (int i=1; i<=dimension; i++)
	{
		if (i!=dimension)
			input >> weights[i] >> dummy;
		else
			input >> weights[i] >> dummy;
	}
}

//! output the prediction value
double 
rlsk_pf::output (const vector <double> &input) const
{
	assert (input.size () == dimension);
	assert (input.size () == (weights.size()-1));

	double x = xzero * weights[0];
	for (int i = 0; i < dimension; i++)
	{
		x += input[i] * weights[i+1];
	}

	if (isnan(x))
	{
		assert(false);
		cout << "NAN - INPUT ";
 		copy (input.begin(), input.end(), ostream_iterator<double>(cout," "));
		cout << " - WEIGHTS ";
 		copy (weights.begin(), weights.end(), ostream_iterator<double>(cout," "));
		cout << endl;
	}
	return x;
};
	
//! latex string representing the prediction function
string 
rlsk_pf::latex_equation(int precision) const
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
rlsk_pf::equation() const
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
rlsk_pf::clear ()
{
	for (int i = 0; i < dimension + 1; i++)
	{
		weights[i] = 0;
	}
}

} //! end xcsflib namespace
