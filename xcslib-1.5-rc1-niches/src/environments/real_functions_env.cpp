#include "real_functions_env.h"
#include "xcs_utility.h"
#include <filesystem>

bool			real_functions_env::init = false;
double			real_functions_env::min_input;
double			real_functions_env::max_input;
unsigned long	real_functions_env::no_inputs;

// [evostar2027 campaign patch] "min input"/"max input" are the keys actually read by set_parameters();
// without them in this list check_parameters() rejects them and the domain is stuck at [0,1].
const std::vector<std::string> real_functions_env::configuration_parameters = {"function", "min value", "max value", "min input", "max input", "scale factor", "sampling resolution","number of segments","random function in every experiment","save functions filename","number of tiles"};

real_functions_env::real_functions_env(xcslib::configuration_manager& xcs_config)
{

	if (!real_functions_env::init)
	{
		if (!xcs_config.exist(tag_name()))
		{
			xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
		}

		xcs_config.check_parameters(tag_name(),configuration_parameters);

		set_parameters(xcs_config);

		print_parameters(clog);

		problem_in.reserve(no_inputs);

		current_inputs.clear();

		for(int i=0;i<no_inputs;i++)
		{
			current_inputs.push_back(min_input);
		}

		real_functions_env::init = true;
	}
}

//! set the parameters from the configuration file
void 
real_functions_env::set_parameters(xcslib::configuration_manager & xcs_config)
{
	string str_selected_function;

	try {
		//! function to be approximated
		str_selected_function = (string) xcs_config.Value(tag_name(), "function");
	} catch (...) {
		xcs_utility::error(class_name(), "constructor", "attribute \'function\' not found in <" + tag_name() + ">", 1);
	}

	//! input range
	min_input = xcs_config.Value(tag_name(), "min input", 0.0);
	max_input = xcs_config.Value(tag_name(), "max input", 1.0);

	//! scale factor that multiplies both input and output (it zooms)
	scale_factor = xcs_config.Value(tag_name(), "scale factor", 1.0);

	sampling_resolution = xcs_config.Value(tag_name(), "sampling resolution", 0.01);

	set_function(xcs_config, str_selected_function, no_inputs);
}

//! print the current parameter setting 
void 
real_functions_env::print_parameters(ostream& OUTPUT)
const
{
	OUTPUT << "<" << tag_name() << ">" << endl;
	OUTPUT << "\t" << "function" << " = " << str_selected_function << endl;
	OUTPUT << "\t" << "min value" << " = " << std::fixed << std::setprecision(5) << min_input << endl;
	OUTPUT << "\t" << "max value" << " = " << std::fixed << std::setprecision(5) << max_input << endl;
	OUTPUT << "\t" << "scale factor" << " = " << std::fixed << std::setprecision(2) << scale_factor << endl;
	OUTPUT << "\t" << "sampling resolution" << " = " << std::fixed << std::setprecision(2) << sampling_resolution << endl;
	OUTPUT << "</" << tag_name() << ">" << endl;
}

void
real_functions_env::begin_problem()
{
	for (int i=0;i<no_inputs;i++)
	{
		current_inputs[i] = xcs_random::random()*(max_input - min_input) + min_input;
	}

    inputs.set_numeric_representation(current_inputs);
}

string real_functions_env::trace() const
{
    ostringstream output_string;
    output_string << current_inputs[0];
    for (int i = 1; i < no_inputs; i++) {
        output_string << ";" << current_inputs[i];
    }
    output_string << ";" << current_reward;
    return output_string.str();
}

void
real_functions_env::reset_input()
{
	vector<double> initial_configuration(no_inputs,min_input);
	problem_in = initial_configuration;

    inputs.set_numeric_representation(initial_configuration);
}

bool
real_functions_env::next_input()
{
	double step_size = sampling_resolution;
	
	bool stop = false;

	problem_in[no_inputs-1]+=step_size;


	for(int i=0;i<=no_inputs;i++)
	{
		if (stop == false)
		{
			if (problem_in[no_inputs-1-i]>max_input)
			{
				if (no_inputs-1-i == 0)
				{
					return false;
				}
				problem_in[no_inputs-1-i]=min_input;
				problem_in[no_inputs-1-i-1]+=step_size;
			}
			else
			{
				stop = true;
			}
		} else {
			current_inputs = problem_in;
			inputs.set_numeric_representation(current_inputs);
			return true;
		}
	}

	return false;
}

void 
real_functions_env::set_function(xcslib::configuration_manager &configuration, string str_fun, unsigned long &no_inputs)
{
	if (str_fun=="sine")
	{ 
		function = &real_functions_env::sine;	no_inputs = 1;
		real_function=t_real_function::SINE;
		return;
	}

	if (str_fun=="f1")
	{ 
		function = &real_functions_env::f1;	no_inputs = 2;
		real_function=t_real_function::F1;
		return;
	}

	if (str_fun=="f2")
	{ 
		function = &real_functions_env::f2;	no_inputs = 2;
		real_function=t_real_function::F2;
		return;
	}

	if (str_fun=="f3")
	{ 
		function = &real_functions_env::f3;	no_inputs = 2;
		real_function=t_real_function::F3;
		return;
	}

	if (str_fun=="frog")
	{ 
		function = &real_functions_env::frog;	no_inputs = 2;
		real_function=t_real_function::FROG;
		return;
	}

	if (str_fun=="step")
	{ 
		function = &real_functions_env::step;	no_inputs = 2;
		real_function=t_real_function::STEP;
		return;
	}

	if (str_fun=="sine3")
	{ 
		function = &real_functions_env::sine3; 	no_inputs = 1;
		real_function=t_real_function::SINE3;
		return;
	}

	// [evostar2027 campaign patch] benchmark functions F4-F7 (see BENCHMARK_FUNCTIONS.md)
	if (str_fun=="sine4")
	{
		function = &real_functions_env::sine4; 	no_inputs = 1;
		real_function=t_real_function::SINE4;
		return;
	}

	if (str_fun=="abs")
	{
		function = &real_functions_env::abs_mix; 	no_inputs = 1;
		real_function=t_real_function::ABS;
		return;
	}

	if (str_fun=="sincos2d")
	{
		function = &real_functions_env::sincos2d; 	no_inputs = 2;
		real_function=t_real_function::SINCOS2D;
		return;
	}

	if (str_fun=="friedman5")
	{
		function = &real_functions_env::friedman5; 	no_inputs = 5;
		real_function=t_real_function::FRIEDMAN5;
		return;
	}

	if (str_fun=="tiling")
	{ 
		function = &real_functions_env::tile; 	no_inputs = 2;
		real_function=t_real_function::TILING2D;

		//! set the specific parameters
		set_parameters_tiling2D(configuration);
		return;
	}

	if (str_fun=="piecewise-linear")
	{ 
		function = &real_functions_env::piecewise_linear;	no_inputs = 1;
		real_function=t_real_function::PIECE_WISE_LINEAR;

		//! set the specific parameters
		set_parameters_piecewise_linear(configuration);
		return;
	}

	// if (str_fun=="Sine4")	
	// { 
	// 	function = &real_functions_env::rf_sin4;
	// 	no_inputs = 1;
	// 	return;
	// }

	// if (str_fun=="Abs")
	// { 
	// 	function = &real_functions_env::rf_abs;
	// 	no_inputs = 1;
	// 	return;
	// }

	// if (str_fun=="Abs2")	
	// { 
	// 	function = &real_functions_env::rf_abs2;
	// 	no_inputs = 1;
	// 	return;
	// }

	// if (str_fun=="Pol")
	// {
	// 	function = &real_functions_env::rf_pol;
	// 	no_inputs = 1;
	// 	return;
	// }

	xcs_utility::error(class_name(), "constructor", "function \'"+str_fun+"\' not found in <" + tag_name() + ">", 1);
}

void real_functions_env::begin_experiment()
{
	switch (real_function)
	{
		case t_real_function::PIECE_WISE_LINEAR:
			begin_experiment_piecewise_linear();
			break;
		case t_real_function::TILING2D:
			begin_experiment_tiling2D();
			break;
		default:
			// other functions does not require a specific initialization
			break;
	}
}


#pragma region TILING2D
void real_functions_env::set_parameters_tiling2D(xcslib::configuration_manager &xcs_config)
{
	tile_function_parameters.no_tiles = (unsigned long) xcs_config.Value(tag_name(), "number of tiles", (unsigned long) 2);
	tile_function_parameters.tile_size = 1/double(tile_function_parameters.no_tiles);

	xcs_utility::set_flag(xcs_config.Value(tag_name(), "random function in every experiment", "off"), tile_function_parameters.use_different_functions_in_every_experiment);

	tile_function_parameters.output_filename = (string) xcs_config.Value(tag_name(), "save functions filename","");

	try {
		std::filesystem::remove(tile_function_parameters.output_filename);
	} catch (const std::filesystem::filesystem_error& err) {
		ostringstream error_message; 
		error_message << "filesystem error: " << err.what();
		xcs_utility::error(class_name(), "set_parameters_tiling2D", error_message.str(), 1);		
	}

	real_function = t_real_function::TILING2D;

	// tile_function_parameters.tile_values.clear();

	// for(int i=0; i<tile_size*tile_size; i++)
	// 	tile_function_parameters.tile_values.push_back(i);
	// std::shuffle(tile_function_parameters.tile_values.begin(),tile_function_parameters.tile_values.end(), xcs_random::rng());

	// for(int i=0; i<tile_function_parameters.tile_size*tile_function_parameters.tile_size; i++)
	// 	cerr << tile_function_parameters.tile_values[i] << " ";
	// cerr << endl;

}

void real_functions_env::begin_experiment_tiling2D()
{
	if (tile_function_parameters.use_different_functions_in_every_experiment ||
		tile_function_parameters.tile_values.size()==0)
	{
		tile_function_parameters.tile_values.clear();

		for(int i=0; i<tile_function_parameters.tile_size*tile_function_parameters.tile_size; i++)
			tile_function_parameters.tile_values.push_back(i);
		std::shuffle(tile_function_parameters.tile_values.begin(),tile_function_parameters.tile_values.end(), xcs_random::rng());

		if (tile_function_parameters.output_filename!="")
		{
			ofstream OUTPUT(tile_function_parameters.output_filename,std::ios::app);
			OUTPUT << ">> tile values = [" << tile_function_parameters.tile_values[0];
			for(int i=1;i<tile_function_parameters.no_tiles;i++)
			{
				OUTPUT << "," << tile_function_parameters.tile_values[i];
			}
			OUTPUT << "] ";		
		}
	}
}

#pragma endregion






#pragma region PIECEWISE_LINEAR
void real_functions_env::set_parameters_piecewise_linear(xcslib::configuration_manager &xcs_config)
{
	piece_wise_linear_function_parameters.no_pieces = (unsigned long) xcs_config.Value(tag_name(), "number of segments", (unsigned long) 5);
	piece_wise_linear_function_parameters.interval = 1/double(piece_wise_linear_function_parameters.no_pieces);
	
    xcs_utility::set_flag(xcs_config.Value(tag_name(), "random function in every experiment", "off"), piece_wise_linear_function_parameters.use_different_functions_in_every_experiment);
	piece_wise_linear_function_parameters.output_filename = (string) xcs_config.Value(tag_name(), "save functions filename","");

	try {
		std::filesystem::remove(piece_wise_linear_function_parameters.output_filename);
	} catch (const std::filesystem::filesystem_error& err) {
		ostringstream error_message; 
		error_message << "filesystem error: " << err.what();
		xcs_utility::error(class_name(), "set_parameters_piecewise_linear", error_message.str(), 1);		
	}

	real_function = t_real_function::PIECE_WISE_LINEAR;	
}

void real_functions_env::begin_experiment_piecewise_linear()
{
	if (piece_wise_linear_function_parameters.use_different_functions_in_every_experiment ||
		(piece_wise_linear_function_parameters.slope.size()==0 && 
		piece_wise_linear_function_parameters.intercept.size()==0))
	{
		piece_wise_linear_function_parameters.slope.clear();
		piece_wise_linear_function_parameters.intercept.clear();	

		// just for brevity in the following code
		double interval = piece_wise_linear_function_parameters.interval;
		int no_pieces = piece_wise_linear_function_parameters.no_pieces;

		vector<double> y;
		for(int i=0;i<no_pieces+1;i++)
		{
			double value = xcs_random::random()*6.0-2;
			y.push_back(double(value));
		}
	
		for(int i=0;i<no_pieces;i++)
		{
			double xs = i*interval;
			double ys = double(y[i]);
	
			double xd = (i+1)*interval;
			double yd = y[i+1];
	
			double sl = (yd-ys)/(xd-xs);
			double in = yd - sl*xd;
			piece_wise_linear_function_parameters.slope.push_back(sl);
			piece_wise_linear_function_parameters.intercept.push_back(in);
	
		}
	
		if (piece_wise_linear_function_parameters.output_filename!="")
		{
			ofstream OUTPUT(piece_wise_linear_function_parameters.output_filename,std::ios::app);
			OUTPUT << ">> slope = [" << piece_wise_linear_function_parameters.slope[0];
			for(int i=1;i<no_pieces;i++)
			{
				OUTPUT << "," << piece_wise_linear_function_parameters.slope[i];
			}
			OUTPUT << "] ";
		
			OUTPUT << "intercept = [" << piece_wise_linear_function_parameters.intercept[0];
			for(int i=1;i<no_pieces;i++)
			{
				OUTPUT << "," << piece_wise_linear_function_parameters.intercept[i];
			}
			OUTPUT << "]" << endl;
		}
	}
}

#pragma endregion