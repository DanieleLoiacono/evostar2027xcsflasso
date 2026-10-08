/*!
 * \file grid_env.cc
 *
 * \brief implements and empty squared room [0,0] to [1,1]
 *
 */
// #define __DEBUG_GRID_ENV__
#include "xcs_random.h"
#include "grid_env.h"

//! 	class not yet initialized
bool	grid_env::init = false;


grid_env::grid_env(xcslib::configuration_manager& xcs_config)
{
	
	if (!grid_env::init)
	{		
		string		str_fixed_start;

		step_size = xcs_config.Value(tag_name(), "step size", 0.05);				//! step size is 0.05 by default
		str_fixed_start = (string) xcs_config.Value(tag_name(), "fixed start", "off");		//! start is random by default
		obstacles_filename = (string) xcs_config.Value(tag_name(), "obstacles file", "");	//! no obstacle by default
		sampling_resolution = xcs_config.Value(tag_name(), "sampling resolution", 0.05);	//! 
		noise_stdev = xcs_config.Value(tag_name(), "noise", 0.0);				//! noise on the actions

		scale = xcs_config.Value(tag_name(), "scale", 1.);				//! no scaling by default;
		shift = xcs_config.Value(tag_name(), "shift", 0.);				//! no shift by default;

		xcs_utility::set_flag(str_fixed_start, flag_fixed_start);
			
		if (obstacles_filename!="") load_obstacles(obstacles_filename);	
		
		grid_env::init = true;
		step_size *= scale;
		sampling_resolution *= scale;
	}
}

void 
grid_env::print_parameters(ostream& OUTPUT) 
const
{
	OUTPUT << "<" << tag_name() << ">" << endl;
	OUTPUT << "step size = " << step_size << endl;
	OUTPUT << "fixed start = " << (flag_fixed_start?"on":"off") << endl;
	OUTPUT << "sampling resolution = " << sampling_resolution << endl;
	OUTPUT << "noise = " << noise_stdev << endl;
	OUTPUT << "scale = " << scale << endl;
	OUTPUT << "shift = " << shift << endl;
	if (obstacles_filename!="")
	{
		OUTPUT << "obstacles file = " << obstacles_filename << endl;
	}
	OUTPUT << "</" << tag_name() << ">" << endl;
}



grid_env::~grid_env()
{
	//! nothing required
}

void
grid_env::begin_problem()
{
	ostringstream	PATH;

	//! at the beginning of the problem the previous information about the agent's path is cleared
	path = "";
	
	if (flag_fixed_start)
	{
		//! fixed start from (0,0)
		x=0;
		y=0;
	} else 	{
		//! start from a random position different from the goal
		do 
		{			
			x = xcs_random::random();
			y = xcs_random::random();			
		} while ((x>=1.0)&&(y>=1.0));
	}

	x = x*scale + shift;
	y = y*scale + shift;

#ifdef __DEBUG_GRID_ENV__
	cerr << "SCALE = " << scale << " SHIFT = " << shift << endl;
	cerr << "CURRENT POSITION = " << x << " " << y << endl;
#endif
	flag_corner_reached = false;

	current_reward = 0;

	set_state();

	PATH << state();
	path = "("+PATH.str()+")";

}

void
grid_env::perform(const t_action& action)
{	
	ostringstream 	PATH;
	static double	inc_x[] = { 0.0, 1.0,  0.0, -1.0};
	static double	inc_y[] = { 1.0, 0.0, -1.0,  0.0};

	double	next_x;
	double	next_y;

	unsigned long act = action.value();

	assert(act<4);

	next_x = x + inc_x[act]*step_size;
	next_y = y + inc_y[act]*step_size;

	if (noise_stdev>0.0)
	{
		  double noise_x = xcs_random::nrandom()*noise_stdev;
		  double noise_y = xcs_random::nrandom()*noise_stdev;
		  
		  next_x += noise_x;
		  next_y += noise_y;		  	
	}

	if ((next_x >= scale+shift) && (next_y >= scale+shift))
		flag_corner_reached = true;
		
	next_x = max(next_x, shift);
	next_x = min(next_x, shift+scale);
	
	next_y = max(next_y, shift);
	next_y = min(next_y, shift+scale);

	
	x = next_x;
	y = next_y;

#ifdef __DEBUG_GRID_ENV__
	cout << "TRUE = (" << next_x << " " << next_y << ")";
	cout << endl;

	cerr << "CURRENT POSITION = " << x << " " << y << endl;
#endif

	if (flag_corner_reached)
		current_reward = 0;
	else 
		current_reward = -0.5 + additional_reward((x-shift)/scale,(y-shift)/scale);		
	
	//cout << "CURRENT REWARD = " << current_reward << endl;

	char c1,c2;
	if (x==1)
		c1='*';
	else
		c1=' ';
	if (y==1)
		c2='*';
	else
		c2=' ';
	set_state();

	PATH << "(" << state() << c1 << c2 <<")";
	path = path +PATH.str();

}

string
grid_env::trace() const
{
	return path;
}

void
grid_env::reset_input()
{
	problem_x = shift;
	problem_y = shift;

	x = shift;
	y = shift;
	set_state();
}

bool
grid_env::next_input()
{
	problem_y+=sampling_resolution;
	if (problem_y>=scale+shift+sampling_resolution)
	{
		problem_y = shift;
		problem_x+=sampling_resolution;
		if (problem_x>=shift+scale+sampling_resolution) return false;
	}

	x = problem_x;
	y = problem_y;

	set_state();

	return true;
}

void
grid_env::save_state(ostream& output) const
{
	output << x << '\t' << y << '\t';	
}

void
grid_env::restore_state(istream& input)
{
	input >> x >> y;	
}

void
grid_env::reset_problem()
{
	reset_input();
}


bool
grid_env::next_problem()
{
	return next_input();
}

void 
grid_env::load_obstacles(const string &fn)
{
	ifstream in(fn.c_str());
	if(!in.good())
	{
		xcs_utility::error(class_name(),"class constructor", "Cannot open obstacles file",1);
	}
	
	obstacle obst;
	obstacles.clear();
	while (in>>obst)
	{
		obstacles.push_back(obst);
		cerr << "One obstacle Loaded: " << obst << endl;
	}
}


double 
grid_env::additional_reward (double x, double y)
{
	double add_rew = 0;
	for (int i=0; i < obstacles.size(); i++)
	{
		if ( (obstacles[i].x1 <= x) && (obstacles[i].x2 >= x) 
				&& (obstacles[i].y1 <= y) && (obstacles[i].y2 >= y) )
		{
			add_rew += obstacles[i].additional_cost;
		}		
	}
	return add_rew;	
}

bool
grid_env::is_terminal(const t_state &state) const
{
	vector<double> coordinates = state.numeric_representation();

	assert(coordinates.size()==2);
	
	return (coordinates[0]>=scale) && (coordinates[1]>=scale);
}