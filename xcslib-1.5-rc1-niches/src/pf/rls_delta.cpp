#include <sstream>
#include <iostream>
#include <iterator>
#include <cassert>
#include <cmath>
#include "xcs_utility.h"
#include "pf/rls_delta.h"

using namespace std;

namespace xcsflib
{

bool	rls_delta_pf::init = false;
double	rls_delta_pf::xzero;
double	rls_delta_pf::delta;

void
rls_delta_pf::reset_matrix()
{
	const unsigned long n = dimension + 1;
	V.assign(n * n, 0.0);
	for (unsigned long i = 0; i < n; i++)
	{
		V[i * n + i] = delta;
	}
}

rls_delta_pf::rls_delta_pf( void * owner ) : base_pf( owner )
{
	assert(init);
	weights.assign(dimension + 1, 0.0);
	reset_matrix();
}

rls_delta_pf::rls_delta_pf(xcslib::configuration_manager& xcs_config)
{
	if (dimension==0)
	{
		xcs_utility::error(class_name(), "constructor", "section <prediction::based> must be inited before <"+tag_name()+">", 1);
	}

	if ( !rls_delta_pf::init )
	{
		if (!xcs_config.exist(tag_name()))
		{
			//! section not present: the function is simply not available
			rls_delta_pf::init = false;
			return;
		}

		try {
			xzero = xcs_config.Value(tag_name(), "x0");
		} catch (...) {
			xcs_utility::error(class_name(), "constructor", "attribute \'x0\' not found in <" + tag_name() + ">", 1);
		}

		try {
			delta = xcs_config.Value(tag_name(), "delta");
		} catch (...) {
			xcs_utility::error(class_name(), "constructor", "attribute \'delta\' not found in <" + tag_name() + ">", 1);
		}

		if (!(delta > 0) || !std::isfinite(delta))
		{
			xcs_utility::error(class_name(), "constructor", "attribute \'delta\' must be finite and > 0", 1);
		}

		clog << "*** rls_delta: x0 = " << xzero << " delta = " << delta << endl;

		rls_delta_pf::init = true;
	}
}

void rls_delta_pf::recombine (base_pf *f)
{
	//! nothing done during recombination (as in rls_pf)
}

base_pf*
rls_delta_pf::clone(void *owner) const
{
	//! offspring: parent's weights, V restarted from delta * I
	rls_delta_pf *f = new rls_delta_pf(owner);
	f->weights = this->weights;
	return f;
}

void
rls_delta_pf::update(const vector <double> &inputs, double target)
{
	vector<double> x;
	if (degree>1)
		polynomial(inputs, x);
	else
		x = inputs;

	assert (x.size() == dimension);

	const unsigned long n = dimension + 1;

	//! phi = [x0, x1, ..., xn]
	vector<double> phi(n);
	phi[0] = xzero;
	for (unsigned long i = 0; i < dimension; i++)
		phi[i+1] = x[i];

	//! a-priori error e = y - w^T phi
	double error = target;
	for (unsigned long i = 0; i < n; i++)
		error -= weights[i] * phi[i];

	//! num = V phi
	vector<double> num(n, 0.0);
	for (unsigned long i = 0; i < n; i++)
		for (unsigned long j = 0; j < n; j++)
			num[i] += V[i * n + j] * phi[j];

	//! beta_rls = 1 + phi^T V phi
	double beta = 1.0;
	for (unsigned long i = 0; i < n; i++)
		beta += phi[i] * num[i];

	//! V <- V - beta^{-1} (V phi)(V phi)^T   (V symmetric: V phi phi^T V = num num^T)
	for (unsigned long i = 0; i < n; i++)
		for (unsigned long j = 0; j < n; j++)
			V[i * n + j] -= num[i] * num[j] / beta;

	//! K = V_t phi = num / beta ;  w <- w + K e
	for (unsigned long i = 0; i < n; i++)
		weights[i] += (num[i] / beta) * error;

	for (unsigned long i = 0; i < n; i++)
	{
		if (!std::isfinite(weights[i]))
		{
			xcs_utility::error(class_name(), "update", "non-finite weights", 1);
		}
	}
}

void
rls_delta_pf::print( ostream & output ) const
{
	output <<"[";
	for (unsigned long i=0; i<=dimension; i++)
	{
		if (i!=dimension)
			output << weights[i] << ";";
		else
			output << weights[i] << "]";
	}
}

void
rls_delta_pf::read(istream &input)
{
	weights.assign(dimension + 1, 0.0);
	char dummy;
	input >> dummy;			//! '['
	for (unsigned long i=0; i<=dimension; i++)
	{
		input >> weights[i] >> dummy;	//! ';' or ']'
	}
	reset_matrix();
}

double
rls_delta_pf::output (const vector <double> &inputs) const
{
	vector<double> x;
	if (degree>1)
		polynomial(inputs, x);
	else
		x = inputs;

	assert (x.size () == dimension);
	assert (x.size () == (weights.size()-1));

	double y = xzero * weights[0];
	for (unsigned long i = 0; i < dimension; i++)
	{
		y += x[i] * weights[i+1];
	}
	assert(!std::isnan(y));
	return y;
};

string
rls_delta_pf::latex_equation(int precision) const
{
	ostringstream str;
	str.setf( ios::fixed );
	str.precision(precision);
	str << weights[0] * xzero;
	for(unsigned long i=1; i<=dimension; i++)
	{
		str << "+" << weights[i] << "\\times" << "x_{" << i << "}";
	}
	return str.str();
}

string
rls_delta_pf::equation() const
{
	ostringstream str;
	str << weights[0] << "*" << xzero;
	for(unsigned long i=1; i<=dimension; i++)
	{
		str << "+" << "x" << i << "*" << weights[i];
	}
	return str.str();
}

void
rls_delta_pf::clear ()
{
	weights.assign(dimension + 1, 0.0);
	reset_matrix();
}

} //! end xcsflib namespace
