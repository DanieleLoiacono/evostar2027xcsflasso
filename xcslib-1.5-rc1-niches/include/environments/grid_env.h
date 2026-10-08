#ifndef __GRID_ENV__
#define __GRID_ENV__

#include <sstream>
#include "rl_definitions.h"
#include "environment_base.h"
#include "configuration_manager.h"

/*!
 * \class grid_env grid_env.hpp
 * \brief implements the 2d Gridworld
 * \sa environment_base
 */

class grid_env : public virtual environment_base<real_inputs,integer_action>
{	

	//! define an obstacle as a rectangular area with an additional negative reward	
	class obstacle 
	{	
		public:
			
		//! coordinates of the top left corner of the obstacles
		double x1;
		double y1;
		
		//! coordinates of the bottom right corner of the obstacles
		double x2;
		double y2;
		
		//! additional cost of passing through obstacles
		double additional_cost;
		
		friend ostream& operator<<(ostream& output, const obstacle& o)
		{
			output << o.x1 << "\t" << o.y1 << "\t" << o.x2 << "\t" << o.y2;
			output << "\t" << o.additional_cost << endl;
			
			// gave the warning otherwise
			return output;
		}
		
		friend istream& operator>>(istream& input, obstacle& o)
		{
			input >> o.x1 >> o.y1 >> o.x2 >> o.y2 >> o.additional_cost;

			// gave the warning otherwise
			return input;
		}
	};
					
public:
	string class_name() const { return string("grid_env"); };

	string tag_name() const { return string("environment::gridworld"); };
	
	//! Class constructor that reads the class parameters through the configuration manager.
	grid_env(xcslib::configuration_manager&);

	void print_parameters(ostream &stream) const;

	//! Class destructor.
	~grid_env();
	
	bool is_single_step() const { return false; }
	
	void begin_problem();
	void end_problem() {};

	void begin_experiment() {};
	void end_experiment() {};

	//! the problem ends when the top corner has been reached
	bool stop() const {return flag_corner_reached;};

	bool is_terminal(const t_state& state) const;	
	
	void perform(const t_action& action);

	string trace() const;

	bool allow_test() const {return true;};

	void reset_problem();
	bool next_problem();

	void reset_input();
	bool next_input();

	void save_state(ostream& output) const;
	void restore_state(istream& input);

	//! true if the problem is single-step or multi-step
	virtual bool single_step() const {return false;};

 	//! return the current reward
	double reward() const {assert(current_reward==grid_env::current_reward); return current_reward;};

	//! return the current state
	t_state state() const { return inputs; };

	//! print the current state to output
	void print(ostream& output) const { output << "(" << x << "," << y << ")\t" << state();};

 private:
	static bool			init;				//!< true if the class has been inited through the configuration manager
	t_state				inputs;				//!< current input configuration
 	vector<obstacle>	obstacles;			//!< obstacles of the 
	string				obstacles_filename;

        //! true if the start position in the environment are set so to visit all the positions the same number of time
	bool		flag_fixed_start;
	
	//! current reward returned for the last action performed
	double		current_reward;

	//! \var step_size size of the action step 
	double 		step_size;

	//! parameters to scale and shift the input domain from the usual (0,0)-(1,1) square
	double 		shift;			//! term added to x and y (0 by default)
	double		scale;			//! term used to scale x and y (1 by default)
	double		noise_stdev;
	
	//! sampling resolution used for testing
	double 		sampling_resolution;

    	//! current position
	double 		x;
	double 		y;
	
	//! problem x and y for testing
	double 		problem_x;
	double 		problem_y;
 
	//! path traces the path the car followed during the problem
	string		path;

	//! true if the goal corner has been reached
	bool		flag_corner_reached;

	void set_state() 
	{
		ostringstream str;
		str << x << " " << y;
		inputs.set_string_value(str.str());
	}

	//! load obstacles file	
	void load_obstacles(const string &fn);

	//! return the additional (negative) reward for position (x,y)
	double additional_reward (double x, double y);
};
#endif
