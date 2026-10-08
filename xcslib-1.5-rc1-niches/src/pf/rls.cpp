#include <sstream>
#include <iostream>
#include <iterator>
#include <cassert>
#include "xcs_utility.h"
#include "xcs_random.h"
#include "pf/rls.h"

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

bool	rls_pf::init = false;
double	rls_pf::xzero;
double	rls_pf::delta;


rls_pf::rls_pf( void * owner ) : base_pf( owner )
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

rls_pf::rls_pf(xcslib::configuration_manager& xcs_config)
{
	if (dimension==0)
	{
		xcs_utility::error(class_name(), "constructor", "section <prediction::based> must be inited before <"+tag_name()+">", 1);			
	}
	
	if ( !rls_pf::init )
	{
		if (!xcs_config.exist(tag_name()))
		{
			// xcs_utility::warning(class_name(), "constructor", "section <" + tag_name() + "> not found");
			rls_pf::init = false;
			return;
		}
		
		try {	string	str_random_weights;

			str_random_weights = (string) xcs_config.Value(tag_name(), "random weights", "off");
			xcs_utility::set_flag(str_random_weights,flag_random_weights);

			xzero = xcs_config.Value(tag_name(), "x0");

#ifdef __DEBUG_LINEAR_WILSON__
			cout << "LR = " << learning_rate << endl;
			cout << "X0 = " << xzero << endl;
#endif

		} catch (const char *attribute) {
			string msg = "attribute \'" + string(attribute) + "\' not found in <" + tag_name() + ">";
			xcs_utility::error(class_name(), "constructor", msg, 1);
		}

		V = gsl_matrix_alloc(dimension+1, dimension+1);
		gsl_matrix_set_identity(V);
		gsl_matrix_scale(V, (double) delta);

		rls_pf::init = true;
	}	
}

void rls_pf::recombine (base_pf *f)
{
	/*! nothing done during recombination
	 *  we tried both mixing the weights and averaging them but the performance appers not to be influenced
	 */	
}

base_pf*
rls_pf::clone(void *owner) const
{
	rls_pf *f = new rls_pf(owner);

	f->weights.clear();
	for (int i=0; i<dimension+1; i++)
	{
		f->weights.push_back(this->weights[i]);
	}
	return f;
}

void 
rls_pf::update(const vector <double> &input, double target)
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
	denominator +=1; // beta = 1 + beta = 1 + phi^T * V * phi

    
	//! compute K(t)
	gsl_vector *K = gsl_vector_alloc(dimension+1);

	gsl_vector_memcpy (K, numerator);	//! K <- numerator
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
	// temp_matrix2 = V * temp_matrix = (1/beta) V * phi * phi^T
	
	//! V <- V - K(t) x^T V
	gsl_matrix_sub (V, KxTV);  

	//! add Q
	gsl_matrix *identity = gsl_matrix_alloc(dimension+1,dimension+1);
	gsl_matrix_set_identity(identity);
	gsl_matrix_add (V, identity);  
	gsl_matrix_free(identity);	//! I matrix

	
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
rls_pf::print( ostream & output ) const
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
rls_pf::read(istream &input)
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
rls_pf::output (const vector <double> &input) const
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
rls_pf::latex_equation(int precision) const
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
rls_pf::equation() const
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
rls_pf::clear ()
{
	for (int i = 0; i < dimension + 1; i++)
	{
		weights[i] = 0;
	}
}

} //! end xcsflib namespace
