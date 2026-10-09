#include "rl_definitions.h"
#include "real_inputs.h"
#include "environment_base.h"
#include "configuration_manager.h"
#include "xcs_random.h"

#ifndef __REAL_FUNCTIONS_ENV__
#define __REAL_FUNCTIONS_ENV__

#include <sstream>
#include <cmath>
#include <cassert>

/*!
 * \class functions_env functions_env.hpp
 * \brief implements various real functions
 * \sa environment_base
 */

class real_functions_env : public virtual environment_base<real_inputs, dummy_action>
{
private:
	static bool				init;			//!< true if the class has been inited through the configuration manager
	static double 			min_input;		//! maximum possible input
	static double 			max_input;		//! minimum possible input
	static unsigned long	no_inputs;		//! number of inputs

	real_inputs				inputs;				//!< current input configuration
	vector<double>			current_inputs;		//!< same information as in "inputs" as a vector of double

	//! current reward returned for the last action performed
	double					current_reward;
	double					scale_factor;				//!< scale factor for the target function	
	double					sampling_resolution;		//!< specify the resolution used to test the learned function
	vector<double>			problem_in;
	string 					str_selected_function;
	unsigned long			arity;

	const static std::vector<std::string> configuration_parameters;

	//! selected function 
	double		(real_functions_env::*function)(const vector<double>& current_inputs);			

	struct {
		public:
			/// @brief number of lineat functions
			int no_pieces;
			/// @brief size of the interval
			double interval;
			/// @brief true if a new function is generated in every experiment
			bool use_different_functions_in_every_experiment;
			/// @brief name of the file where all the generated functions are saved
			string output_filename;

			/// @brief linear functions slopes and intercepts
			vector<double> slope;
			vector<double> intercept;
	} piece_wise_linear_function_parameters;

	struct {
		public:
			/// @brief number of lineat functions
			int no_tiles;
			/// @brief size of the interval
			double tile_size;
			/// @brief true if a new function is generated in every experiment
			bool use_different_functions_in_every_experiment;
			/// @brief name of the file where all the generated functions are saved
			string output_filename;
			/// @brief values of the tile steps 
			vector<double> tile_values;
	} tile_function_parameters;

public:
	enum class t_real_function {
		SINE,
		PIECE_WISE_LINEAR,
		TILING2D,
		F1,
		F2,
		F3,
		FROG,
		STEP,
		SINE3,
		SINE4,		// [evostar2027 campaign patch]
		ABS,		// [evostar2027 campaign patch]
		SINCOS2D,	// [evostar2027 campaign patch]
		FRIEDMAN5	// [evostar2027 campaign patch]
	};

	t_real_function real_function = t_real_function::PIECE_WISE_LINEAR;

	string class_name() const { return string("real_functions_env"); };

	string tag_name() const { return string("environment::real_functions"); };

	real_functions_env(xcslib::configuration_manager&);

	//! set the parameters from the configuration file
	void set_parameters(xcslib::configuration_manager & xcs_config);

	//! print the current parameter setting 
	void print_parameters(ostream&) const;

	//! Destructor for the sine environment class.
	~real_functions_env() {};

	bool is_single_step() const { return true; }
	
	void begin_problem();
	void end_problem() {};

	void begin_experiment();
	void end_experiment() {};

	bool stop() const {return true;};

	bool is_terminal(const t_state &state) const { return true; };	
	
	void perform(const t_action& action)
	{
		current_reward = (*this.*function)(current_inputs);
	}

	string trace() const;

	bool allow_test() const {return true;};

	void reset_input();
	bool next_input();

	void save_state(ostream& output) const {};
	void restore_state(istream& input) {};

	//! indicates that sine environments are single step problems
	virtual bool single_step() const {return true;};

	virtual double reward() const {assert(current_reward==real_functions_env::current_reward); return current_reward;};

	virtual real_inputs state() const {return inputs;};

	// virtual void sereal_inputs(real_inputs sensor) {
	// 	this->inputs = sensor;
	// 	current_inputs.clear();
	// 	int i;
	// 	for(i=0;i<current_inputs.size();i++)
	// 		current_inputs.push_back(sensor.input(i));
	// };

	virtual void print(ostream& output) const { output << inputs.string_value();};

	void set_function(xcslib::configuration_manager &configuration, string str_fun, unsigned long &no_inputs);

private:
	double sine(const std::vector <double>& current_inputs)
	{
		assert(current_inputs.size()==1); 
		return double(scale_factor)*sin((2*M_PI*current_inputs[0])/double(scale_factor));
	}

	double f1(const std::vector <double>& current_inputs)
	{
		assert(current_inputs.size()==2); 
		return double(long(current_inputs[0]*3)%3)/3. + double(long(current_inputs[1]*3)%3)/3.;
	}

	double f2(const std::vector <double>& current_inputs)
	{
		assert(current_inputs.size()==2); 
		return double(long((current_inputs[0]+current_inputs[1])*2)%4)/6.;
	}

	double f3(const std::vector <double>& current_inputs)
	{
		assert(current_inputs.size()==2); 
		return sin(2*M_PI*(current_inputs[0]+current_inputs[1]));
	}

	double frog(const std::vector <double>& current_inputs)
	{
		assert(current_inputs.size()==2); 

		double result;
		if (current_inputs[0]+current_inputs[1]<=1)
			result = current_inputs[0]+current_inputs[1];
		if (current_inputs[0]+current_inputs[1]>1)
			result = 2-(current_inputs[0]+current_inputs[1]);
		return result;
	}

	double hill(const std::vector <double>& current_inputs)
	{
		assert(current_inputs.size()==2);
		double result;
		
		// f(x,y) = ((x-0.5)**2+(y-0.5)**2>=0.09) ? 0.28 : ((x-0.5)**2+(y-0.5)**2<0.09) ? (-8*((x-0.5)**2 + (y-0.5)**2))+1 : 1/0
		if (pow((current_inputs[0]-0.5),2)+pow((current_inputs[1]-0.5),2)>=0.09)
			result = 0.28;
		if (pow((current_inputs[0]-0.5),2)+pow((current_inputs[1]-0.5),2)<0.09)
			result = (-8*(pow((current_inputs[0]-0.5),2)+pow((current_inputs[1]-0.5),2)))+1;
		return result;

	}

	double step(const std::vector <double>& current_inputs)
	{
		assert(current_inputs.size()==2); 
		double result;

		if (current_inputs[0]+current_inputs[1]>=1)
			result = 1;
		if (current_inputs[0]+current_inputs[1]<1)
			result = 0;
		return result;
	}

	double	sine3(const vector<double>& current_inputs) 
	{
		assert(current_inputs.size()==1);
		return 	scale_factor*(std::sin((2*M_PI*current_inputs[0])/scale_factor)+std::sin((4*M_PI*current_inputs[0])/scale_factor)+
			std::sin((6*M_PI*current_inputs[0])/scale_factor));
	}

	// [evostar2027 campaign patch] benchmark functions F4-F7 (see BENCHMARK_FUNCTIONS.md)
	double	sine4(const vector<double>& current_inputs)
	{
		assert(current_inputs.size()==1);
		return 	scale_factor*(std::sin((2*M_PI*current_inputs[0])/scale_factor)+std::sin((4*M_PI*current_inputs[0])/scale_factor)+
			std::sin((6*M_PI*current_inputs[0])/scale_factor)+std::sin((8*M_PI*current_inputs[0])/scale_factor));
	}

	double	abs_mix(const vector<double>& current_inputs)
	{
		assert(current_inputs.size()==1);
		return 	scale_factor*std::fabs(std::sin((2*M_PI*current_inputs[0])/scale_factor)+std::fabs(std::cos((2*M_PI*current_inputs[0])/scale_factor)));
	}

	double	sincos2d(const vector<double>& current_inputs)
	{
		assert(current_inputs.size()==2);
		return 	scale_factor*std::sin(2*M_PI*current_inputs[0])*std::cos(2*M_PI*current_inputs[1]);
	}

	double	friedman5(const vector<double>& current_inputs)
	{
		assert(current_inputs.size()==5);
		return 	10*std::sin(M_PI*current_inputs[0]*current_inputs[1]) + 20*std::pow(current_inputs[2]-0.5,2) + 10*current_inputs[3] + 5*current_inputs[4];
	}

	double	tile(const vector<double>& current_inputs) 
	{
		assert(current_inputs.size()==2);
		int xi = (int) (current_inputs[0]/tile_function_parameters.tile_size);
		int yi = (int) (current_inputs[1]/tile_function_parameters.tile_size);

		int p = xi*tile_function_parameters.no_tiles+yi;

		return tile_function_parameters.tile_values[p];
	}

	void set_parameters_tiling2D(xcslib::configuration_manager & xcs_config);
	void begin_experiment_tiling2D();


	//! set the parameters from the configuration file
	void set_parameters_piecewise_linear(xcslib::configuration_manager & xcs_config);
	void begin_experiment_piecewise_linear();

	double	piecewise_linear(const vector<double>& current_inputs) 
	{
		assert(current_inputs.size()==1);

		double x = current_inputs[0];
		int function_index = int(x/piece_wise_linear_function_parameters.interval);
		return x*piece_wise_linear_function_parameters.slope[function_index]+piece_wise_linear_function_parameters.intercept[function_index];
	}
};
#endif
