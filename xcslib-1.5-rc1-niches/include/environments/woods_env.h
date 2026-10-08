#ifndef __WOODS_ENV__
#define __WOODS_ENV__

#include <map>
#include <sstream>

#include "environment_base.h"
#include "binary_inputs.h"
#include "binary_action.h"
#include "configuration_manager.h"

#include "rl_definitions.h"

/*!
 * \class woods_env woods_env.h
 * \brief implements the methods specific for woods environments
 * \sa environment_base
 */

class woods_env : public virtual environment_base<binary_inputs,binary_action>
{
public:
	string class_name() const { return string("woods_env"); };
	string tag_name() const { return string("environment::woods"); };
	
	//! Constructor for the woods environment class. It reads the class parameters through the configuration manager.
	/*!
	 *  This is the only constructor that can be used. 
	 */
	woods_env(xcslib::configuration_manager&);

	//! Destructor for the woods environment class.
	~woods_env();
	
	void begin_problem();
	void end_problem() {};

	void begin_experiment() {};
	void end_experiment() {};

	bool stop() const;
	
	void perform(const t_action& action);

	string trace() const;

	bool allow_test() const {return true;};
	void reset_input();
	bool next_input();

	void reset_problem() { reset_input(); };
	bool next_problem() { return next_input(); };

	void save_state(ostream& output) const;
	void restore_state(istream& input);

	void set_parameters(xcslib::configuration_manager& xcs_config);
	void print_parameters(ostream&) const;

	//! indicates that woods environments are multiple step problems
	virtual bool is_single_step() const {return false;};

	bool is_terminal(const t_state &state) const;

 public:
	virtual double reward() const {assert(current_reward==woods_env::current_reward); return current_reward;};
	virtual t_state state() const { return inputs; };
	virtual void print(ostream& output) const { output << "(" << current_pos_x << "," << current_pos_y << ")\t" << state();};

 private:

	//! agent's position on the grid
	struct GridPosition
	{
		public:
			unsigned long x;
			unsigned long y;
			GridPosition(unsigned long x=0, unsigned long y=0) : x(x), y(y) {}
	};


	//! computes the current a value on a circular scale given its upper limit 
	inline int	cycle(int op, int limit) const;

	//! computes the sensory inputs that are returned in position <x,y>
	void		get_input(unsigned long x, unsigned long y, t_state& sensors) const;

	//! given the current <x,y> position sets the current input and the current reward \sa current_position_x \sa current_position_y
	inline void	set_state();

	//! return true if the position <x,y> is free (i.e., it contains ".")
	inline bool	is_free(unsigned long x, unsigned long y) const;
	// inline bool is_free(const GridPosition &position) const;

	//! return true if the position <x,y> contains food (i.e., "F")
	inline bool	is_food(unsigned long x, unsigned long y) const;
	// inline bool is_food(const GridPosition &position) const;

	//! return true if the position <x,y> contains food (i.e., "F")
	void binary_encode(const string&, string&) const;
	void binary_encode_woods1(const string&, string&) const;
	void binary_encode_woods2(const string &input, string &binary) const;

	//! reads the map
	void read_map(string filename);
	void init_map_data();
	
	static bool		init;			//!< true if the class has been inited through the configuration manager
	t_state			inputs;			//!< current input configuration

    //! true if the start position in the environment are set so to visit all the positions the same number of time
	bool			uniform_start;
	
    //! current reward returned for the last action performed
	double			current_reward;

	//! \var current_configuration index of the current agent's input
	/*!
	 * it is used when scanning all the possible environment configurations with 
	 * the functions \fn reset_input and \fn next_input
	 * \sa reset_input
	 * \sa next_input
	 */
	////unsigned long	current_configuration;			// counter for uniform problem start

	string map_filename;

	//! \var current_state current agent's input
	//unsigned long	current_state;				// counter for the scan of states

	//! \var all the possible input configurations
	vector<string>	configurations;

	//! all possible input configurations when in a goal state
	vector<string> 	goal_input_configurations;


	GridPosition			current_position;
	vector<GridPosition>	free_positions;

    //! current x,y position in the environment
	unsigned long 	current_pos_x;
	unsigned long 	current_pos_y;

 	//! \var flag_binary_sensors specifies whether binary inputs (i.e., 00100...) or symbolic inputs (i.e., "TF......T.") should be returned
	bool			flag_binary_sensors;

	//! \var prob_slide specifies the probability that the agent can slip while it moves (Colombetti and Lanzi 1999)
	double			prob_slide; 

    //! \var env_rows number of rows
	// unsigned long	env_rows;
	unsigned long	no_rows;

    //! \var env_columns number of columns
	// unsigned long	env_columns;
	unsigned long	no_columns;

    //! \var map environment map 
	vector<string>		map;

    //! \var env_free_pos number of free positions (i.e., ".") in the environment
	unsigned long 		no_free_positions;

    //! x,y coordinates of the free positions in the environment
	vector<unsigned long>	free_pos_x;
	//! y coordinates of the free positions in the environment
	vector<unsigned long>	free_pos_y;

	//! path traces the path the agent followed during the problem
	string			path;

	//!  supported parameters in the configuration file
	const static std::vector<std::string> supported_configuration_parameters;

	//! true if the woods2 environment is used
	bool flag_use_woods2_symbols;

	//! environment maps
	std::map<std::string,std::vector<std::string>> binary_environment_maps 
	{
		{"Woods1",
			{
			".....",
			".....",
			"TTF..",
			"TTT..",
			"TTT.."
			}
		},

		{"Woods2",
			{
				"..............................",
				".QQF..QQF..OQF..QQG..OQG..OQF.",
				".OOO..QOO..OQO..OOQ..QQO..QQQ.",
				".OOQ..OQQ..OQQ..QQO..OOO..QQO.",
				"..............................",
				"..............................",
				".QOF..QOG..QOF..OOF..OOG..QOG.",
				".QQO..QOO..OOO..OQO..QQO..QOO.",
				".QQQ..OOO..OQO..QOQ..QOQ..OQO.",
				"..............................",
				"..............................",
				".QOG..QOF..OOG..OQF..OOG..OOF.",
				".OOQ..OQQ..QQO..OQQ..QQO..OQQ.",
				".QQO..OOO..OQO..OOQ..OQQ..QQQ.",
				"..............................",			
			}
		},

		{"Maze4",
			{
				"TTTTTTTT",
				"T..T..FT",
				"TT..T..T",
				"TT.T..TT",
				"T......T",
				"TT.T...T",
				"T....T.T",
				"TTTTTTTT",		
			}
		},

		{"Maze5",
			{
				"TTTTTTTTT",
				"T......FT",
				"T..T.TT.T",
				"T.T.....T",
				"T...TT..T",
				"T.T.T..TT",
				"T.T..T..T",
				"T.....T.T",
				"TTTTTTTTT",		
			}
		},

		{"Maze6",
			{
				"TTTTTTTTT",
				"T.....TFT",
				"T..T.TT.T",
				"T.T.....T",
				"T...TT..T",
				"T.T.T..TT",
				"T.T.....T",
				"T.....T.T",
				"TTTTTTTTT",		
			}
		},

		{"Woods14",
			{
				"TTTTTTTTTTTTT",
				"TT...TTTT.TT.",
				"T.TTT.TT.T.T.",
				"T.TTT.T.TTT.T",
				"TFTTT.TT.TTTT",
				"TTTTTT..TTTTT",
			}
		}

	};

};
#endif
