/*!
 * \class real_interval_condition real_interval_condition.h
 *
 * \brief class for real-interval conditions
 */

#include <string>
#include <sstream>
#include "rl_definitions.h"
#include "configuration_manager.h"
#include "real_inputs.h"
#include "interval.h"
#include "condition_base.h"

#ifndef __REAL_INTERVAL_CONDITION__
#define __REAL_INTERVAL_CONDITION__


#define __CONDITION_VERSION__ "real interval condition"


// using namespace xcslib;

class real_interval_condition : public virtual condition_base<real_interval_condition, real_inputs>
{

public:
	enum class t_mutation_type
	{
		proportional,
		fixed,
		gaussian
	};

	enum class  t_crossover_type
	{
		one_point,
		two_point,
		uniform
	};

private:

	vector< xcslib::interval< double > > value;

	static bool				init;			//!< true if the class has been already inited through the configuration manager
	static unsigned long 	no_inputs;
	static double 			min_input;
	static double 			max_input;
	static double 			r0;
	static double 			m0;
	static bool 			bounded;		//!< if true -> intervals are bounded between min_input and max_input

   	static t_mutation_type	mutation_type;
	static t_crossover_type	crossover_type;

public:
	//! name of the class that implements the condition
	string class_name() const { return string("real_interval_condition"); };

	//! tag used to access the configuration file
	string tag_name() const { return string("condition::real_interval"); };

	//! return the condition as a string
	string string_value() const;

	//! set the condition to a value represented as a string
	void set_string_value(const string& str);

	//! return the condition size
	unsigned long size() const {return no_inputs;};

	//
	//	class constructors
	//

	//! Constructor for the interval condition class that read the class parameters through the configuration manager
	/*!
	 *  This is the first constructor that must be used. Otherwise an error is returned.
	 */
	real_interval_condition(xcslib::configuration_manager&);
	real_interval_condition(unsigned long no_inputs_, double min_input_, double max_input_, double r0_, double m0_, 
		bool bounded_, t_mutation_type mutation_type_, t_crossover_type crossover_type_);

	void print_parameters(ostream& OUTPUT) const;

	//! Default constructor for the interval condition class
	/*!
	 *  If the class parameters have not yet initialized through the configuration manager,
	 *  an error is returned and the method exists to shell.
	 *  \sa real_interval_condition(xcs_config_mgr&)
	 *  \sa xcs_config_mgr
	 */

	real_interval_condition();

	real_interval_condition(const real_interval_condition& cond);

	// real_interval_condition(unsigned long d, double min_in, double max_in, double r=10, double  m=20);

	//
	//	class destructor
	//

	//! Default destructor for the interval condition class
	~real_interval_condition() { value.clear(); };

	//
	//	comparison operators
	//

	//! less than operator
	bool operator< (const real_interval_condition& condition) const;

	//! equality operator
	bool operator==(const real_interval_condition& condition) const;

	//! inequality operator
	bool operator!=(const real_interval_condition& condition) const;

	//
	//	assignment operators
	//

	//! assignment operator for a constant value
	real_interval_condition& operator=(const real_interval_condition& condition);

	// normalizza l'input
	void normalize(vector<double>& inputs, vector<double>& normalized_inputs);
      
	//! return true if the condition matches the input configuration
	bool match(const real_inputs& inputs) const;

	//! set the condition to cover the input
	void cover(const real_inputs& inputs);

	//! mutate the condition according to the mutation rate \emph mu
	void mutate(double mutation_probability);

	//! mutate the condition according to the mutation rate \emph mu; the mutating bits are set according to the current input
	void mutate(double mutation_probability, const real_inputs &inputs);

	//! recombine the condition according to the strategy specified by the method variable
	// \param method the crossover type to be used
	void recombine(real_interval_condition& condition);
	// void recombine(real_interval_condition& condition, unsigned long method);

	//! pretty print the condition to the output stream "output".
	void print(ostream& output) const { output << string_value(); };

	//! true if the representation allow the use of GA subsumption
	virtual bool allow_ga_subsumption() const {return true;};

	//! true if the representation allow the use of action set subsumption
	virtual bool allow_as_subsumption() const {return true;};

	//! return true if the condition is more general than (i.e., subsumes) \emph cond
	bool is_more_general_than(const real_interval_condition& cond) const;
	
	double generality() const 
	{
		double delta = max_input-min_input+1;
		double gen = 0;
		int sz=0;
		for (int i=0; i< value.size(); i++)
		{
			gen += (value[i].get_upper_bound()-value[i].get_lower_bound() + 1);
			sz++;
		}
		return (gen/(sz*delta));
	};

	//! generate a random condition
	void random();
	
	//! additional functions 
	double	min() const {return min_input;};
	double	max() const {return max_input;};
	double	lower(long i) const {assert(i<size()); return value[i].get_lower_bound();};
	double	upper(long i) const {assert(i<size()); return value[i].get_upper_bound();};
	long	condition_size() const {return value.size();};
		
	void single_point_crossover(real_interval_condition& condition);
	void two_points_crossover(real_interval_condition& condition);
	void uniform_crossover(real_interval_condition& condition);

	void fixed_mutation(double mutation_probability);
	
	void gaussian_mutation(double mutation_probability);

	void proportional_mutation(double mutation_probability);

	void set_crossover_type(string str) const;

	void set_mutation_type(string str) const;

	void check(xcslib::interval<double> &it);

	void set_parameters(xcslib::configuration_manager& configuration);

};

#endif
