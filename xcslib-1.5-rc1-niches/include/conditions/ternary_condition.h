/*!
 * \class ternary_condition ternary_condition.h
 *
 * \brief definition class for classifier ternary conditions
 */

#ifndef __TERNARY_CONDITION__
#define __TERNARY_CONDITION__

#define __CONDITION_VERSION__ "ternary strings {0,1,#}"

#include <string>
#include "rl_definitions.h"
#include "configuration_manager.h"
#include "condition_base.h"
#include "binary_inputs.h"

class ternary_condition : public virtual condition_base<ternary_condition, binary_inputs> 
{
public:
	static const char	dont_care;							//! don't care symbol used in conditions

private:
	static bool				init;							//!< true if the class has been already inited through the configuration manager

	string					bitstring;						//!< condition string
	static unsigned long	no_bits;						//!< number of bits in condition
	static double			dont_care_prob;					//!< probability of having a don't care in a random condition
	static bool				flag_mutation_with_dontcare;	//!< true if #s are used in mutation
	static unsigned long	crossover_type;					//!< default crossover type (0=uniform, 1=one-point, 2=two-points)
	static unsigned long	mutation_type;					//!< default mutation type (1=niche mutation, 2=two-values, 3=pure or three values)
	const static vector<string>	configuration_parameters;		//!< accepted parameters

public:
	//! name of the class that implements the condition
	string class_name() const { return string("ternary_condition"); };
	
	//! tag used to access the configuration file
	string tag_name() const { return string("condition::ternary"); };

	//! return the condition as a string
	string string_value() const {return bitstring;};

	//! set the condition to a value represented as a string
	void set_string_value(const string& string_value) {bitstring = string_value;};

	//! return the condition size
	unsigned long size() const {return bitstring.size();};

	//! Constructor for the ternary condition class that read the class parameters through the configuration manager
	/*!
	 *  This is the first constructor that must be used. Otherwise an error is returned.
	 */
	ternary_condition(xcslib::configuration_manager&);

	//! Default constructor for the ternary condition class
	/*!
	 *  If the class parameters have not yet initialized through the configuration manager, 
	 *  an error is returned and the method exists to shell. 
	 *  \sa ternary_condition(xcs_config_mgr&)
	 *  \sa xcs_config_mgr
	 */
	ternary_condition();

	//! Default destructor for the ternary condition class
	~ternary_condition();
	
	//! less than operator
	bool operator< (const ternary_condition& condition) const;

	//! equality operator
	bool operator==(const ternary_condition& condition) const;

	//! inequality operator
	bool operator!=(const ternary_condition& condition) const;

	//! assignment operator for a constant value
	ternary_condition& operator=(const ternary_condition& condition);

	//! return true if the condition matches the input configuration
	bool match(const binary_inputs& input) const;

	//! set the condition to cover the input 
	void cover(const binary_inputs& input);

	//! mutate the condition according to the mutation rate \emph mu
	void mutate(double mutation_rate);

	//! mutate the condition according to the mutation rate \emph mu; the mutating bits are set according to the current input
	void mutate(double mutation_rate, const binary_inputs &inputs);

	//! recombine the condition 
	void recombine(ternary_condition& condition) { recombine(condition,crossover_type); };

	//! pretty print the condition to the output stream "output".
	void print(ostream& output) const { output << bitstring; };

	//! true if the representation allow the use of GA subsumption
	virtual  bool allow_ga_subsumption() const {return true;};

	//! true if the representation allow the use of action set subsumption
	virtual bool allow_as_subsumption() const {return true;};

	//! return true if the condition is more general than (i.e., subsumes) \emph cond
	bool is_more_general_than(const ternary_condition& cond) const;

	//! generate a random condition
	void random();

	//! recombine the condition according to the strategy specified by the method variable
	// \param method the crossover type to be used
	void recombine(ternary_condition& condition, unsigned long method);

	/// @brief  printthe parameters read from the configuration file
	/// @param xcs_config 
	void print_parameters(ostream& OUTPUT) const;

 private:
	/// single point crossover
	void	single_point_crossover(ternary_condition& condition);
 
	/// two point crossover
	void	two_point_crossover(ternary_condition& condition);
 
	/// uniform crossover
	void	uniform_crossover(ternary_condition& condition);

 public:
	//! generality
	double generality() const;

	//! specificity
	double specificity() const;

	/// @brief set the parameters read from the configuration file
	/// @param xcs_config 
	void set_parameters(xcslib::configuration_manager &xcs_config);

};
#endif
