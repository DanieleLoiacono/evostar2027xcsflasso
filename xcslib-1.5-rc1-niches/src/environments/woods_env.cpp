/*!
 * \file woods_env.cpp
 *
 * \brief implementation of woods environments as defined by Wilson (1995)
 *
 * The environment is defined through a map contaning obstacles (T, Q, or O), free positions (.), 
 * and goal positions (F, G). The top left position of the map corresponds to position (0,0).
 *
 * At the beginning of a problem the agent is place at a random position; 
 * the problem ends when the agent reaches a goal position.
 *
 * The configuration file specifies two main information: 
 * - the name of the file that contains the enviroment map and 
 * - the agent sliding probability 
 *
 */

#include "woods_env.h"

bool woods_env::init = false;
const std::vector<std::string> woods_env::supported_configuration_parameters = {"map","binary sensors","slide probability","woods2 sensors"};

woods_env::woods_env(xcslib::configuration_manager& xcs_config)
{
	if (!woods_env::init)
	{
		if (!xcs_config.exist(tag_name()))
		{
			xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
		}
	

		xcs_config.check_parameters(tag_name(), supported_configuration_parameters);

		set_parameters(xcs_config);
		
		set_state();

	}
	woods_env::init = true;
}

woods_env::~woods_env()
{
	map.clear();
}

void
woods_env::read_map(string filename)
{
	/// initialize the map as a vector of strings
	if (binary_environment_maps.count(filename))
	{
		map = binary_environment_maps[filename];
	} else {
		///	read the woods map
		ifstream MAP(filename.c_str());

		/// map not found
		if (!MAP.good())
		{
			/// error: map not found
			xcs_utility::error(class_name(),"class constructor", "map file not found", 1);
		}

		string			row;
		
		map.clear();
		while (MAP >> row)
		{		
			map.push_back(row);	
		}
		MAP.close();

	}
}

void 
woods_env::init_map_data()
{
	no_rows = 0;
	no_columns = 0;
	no_free_positions = 0;

	no_rows = map.size();
	no_columns = map[0].size();

	for(size_t row_i=0; row_i<map.size(); row_i++)
	{
		if (map[row_i].size()!=no_columns)
		{
			xcs_utility::error(class_name(),"class constructor", "map is not rectangular", 1);
		}
	}

	free_pos_x.clear();
	free_pos_y.clear();
	free_positions.clear();

	t_state		free_position;
	t_state		food_position;

	configurations.clear();
	goal_input_configurations.clear();
	no_free_positions = 0;

	for (size_t y=0; y<no_rows; y++)
	{
		for (size_t x=0; x<no_columns; x++)
		{
			if (is_free(x,y))
			{
				get_input(x,y,free_position);
				configurations.push_back(free_position.string_value());

				free_pos_x.push_back(x);
				free_pos_y.push_back(y);

				free_positions.push_back(GridPosition(x,y));

				no_free_positions++;
			}

			if (is_food(x,y))
			{
				get_input(x,y,food_position);
				goal_input_configurations.push_back(food_position.string_value());
			}
		}
	}

	current_pos_x = 0;
	current_pos_y = 0;
	current_position = GridPosition(0,0);

	current_configuration = 0;
}

void 
woods_env::set_parameters(xcslib::configuration_manager& xcs_config)
{
	try {
		map_filename = (string) xcs_config.Value(tag_name(), "map");
	} catch (...)
	{
		string msg = "attribute \'map\' not found in <" + tag_name() + ">";
		xcs_utility::error(class_name(), "constructor", "attribute \'map\' not found in <" + tag_name() + ">", 1);
	}

	xcs_utility::set_flag(xcs_config.Value(tag_name(), "binary sensors", "on"), flag_binary_sensors);

	xcs_utility::set_flag(xcs_config.Value(tag_name(), "woods2 sensors", "off"), flag_use_woods2_symbols);

	prob_slide = xcs_config.Value(tag_name(), "slide probability", 0.0);

	if (prob_slide<0.0 || prob_slide>1.0)
	{
		xcs_utility::error(class_name(), "constructor", "attribute \'slide probability\' must have values in [0.0,1.0]", 1);
	}	

	read_map(map_filename);
	init_map_data();
}

void 
woods_env::print_parameters(ostream& OUTPUT) const
{
	OUTPUT << "<" << tag_name() << ">" << endl;
	OUTPUT << "\t" << "map = " << map_filename << endl;
	OUTPUT << "\t" << "binary sensors = " << (flag_binary_sensors?"on":"off") << endl;
	OUTPUT << "\t" << "sliding probability = " << setprecision(2) << prob_slide << endl;
	OUTPUT << "</" << tag_name() << ">" << endl;
}

void	
woods_env::begin_problem()
{
	ostringstream	PATH;
   	unsigned long	where;

	//! at the beginning of the problem the previous information about the agent's path is cleared
	path = "";

	//! random restart
	where = (unsigned long) (no_free_positions*xcs_random::random());
		
	current_pos_x = free_pos_x[where];
	current_pos_y = free_pos_y[where];

	current_position = free_positions[where];
	assert(current_position.x==current_pos_x);
	assert(current_position.y==current_pos_y);

	set_state();

	PATH << "(" << current_pos_x << "," << current_pos_y << ")";
	path += PATH.str();
}

bool
woods_env::stop()
const
{
	return(is_food(current_pos_x,current_pos_y)); 
}

bool 
woods_env::is_terminal(const t_state &state)
const
{
    vector<string>::const_iterator pos = find(goal_input_configurations.cbegin(),goal_input_configurations.cend(), state.string_value());
	return pos!=goal_input_configurations.cend();
}


void	
woods_env::perform(const t_action& action)
{
	static int 	sliding[] = {-1,1};
	static int	inc_x[] = { 0, +1, +1, +1,  0, -1, -1, -1};
	static int	inc_y[] = {-1, -1,  0, +1, +1, +1,  0, -1};

	unsigned long	act;
	unsigned long	next_x;
	unsigned long	next_y;

	act = action.value();

	if (xcs_random::random()<prob_slide)
	{
		act = cycle(act + sliding[xcs_random::dice(2)], action.actions());
	}

	next_x=cycle(current_pos_x+inc_x[act], no_columns);
	next_y=cycle(current_pos_y+inc_y[act], no_rows);

	if ((is_free(next_x,next_y)) || (is_food(next_x,next_y)))
	{
		current_pos_x = next_x;
		current_pos_y = next_y;
		set_state();
	}
	ostringstream PATH;
	PATH << "(" << current_pos_x << "," << current_pos_y << ")";
	path += PATH.str();
}

string
woods_env::trace() const
{
	return path;
}

void
woods_env::save_state(ostream& output) const
{
	output << endl;
	output << current_pos_x << '\t' << current_pos_y << '\t';
	output << current_configuration << endl;
}

void
woods_env::restore_state(istream& input)
{
	input >> current_pos_x; 
	input >> current_pos_y;
	input >> current_configuration;
	set_state();
}

inline 
int	
woods_env::cycle(int op, int limit) 
const
{
	return ((op % limit)>=0)?(op % limit):((op % limit) + limit);
}

inline
void	
woods_env::get_input(unsigned long x,unsigned long y, t_state& inputs)
const
{
	static int	inc_x[] = { 0, +1, +1, +1,  0, -1, -1, -1};
	static int	inc_y[] = {-1, -1,  0, +1, +1, +1,  0, -1};
	unsigned long	sx,sy;
	unsigned long	pos;

	string		symbolic_input = "";
	string		binary_input = "";

	for(pos=0; pos<8; pos++)
	{
		sx = cycle(x+inc_x[pos], no_columns);
		sy = cycle(y+inc_y[pos], no_rows);
		symbolic_input += map[sy][sx];
	}

	if (flag_binary_sensors)
	{
		string	binary_input;
		binary_encode(symbolic_input, binary_input);		
		inputs.set_string_value(binary_input);

	} else {
		inputs.set_string_value(symbolic_input);
	}

}

void
woods_env::binary_encode_woods2(const string &input, string &binary)
const
{
	binary = "";

	for(string::const_iterator pos=input.begin(); pos!=input.end(); pos++)
	{
		switch (*pos)
		{
			case 'O': // rock
				binary += "010";
				break;
			case 'Q': // rock
				binary += "011";
				break;

			case 'F': // food of type 'F'
				binary += "110";
				break;
			case 'G': // food of type 'G'
				binary += "111";
				break;
			case '.': // free 
				binary += "000";
				break;
			default:
				xcs_utility::error(class_name(),"class constructor", "unrecognized map symbol", 1);
		}
	}
}
inline
void	
woods_env::set_state()
{
	get_input(current_pos_x,current_pos_y,inputs);

	if (is_food(current_pos_x,current_pos_y))
	{
		current_reward = 1000;
	} else {
		current_reward = 0;
	}
}

inline
bool 
woods_env::is_free(unsigned long x, unsigned long y)
const
{
	return(map[y][x]=='.');
}

inline
bool 
woods_env::is_food(unsigned long x, unsigned long y)
const
{
	if (flag_use_woods2_symbols)
		return ( (map[y][x]=='F') || (map[y][x]=='G') );
	else 
		return (map[y][x]=='F');
}

void
woods_env::binary_encode(const string &input, string &binary)
const
{
	if (flag_use_woods2_symbols)
		binary_encode_woods2(input, binary);
	else
		binary_encode_woods1(input, binary);
}

void
woods_env::binary_encode_woods1(const string &input, string &binary)
const
{
	binary = "";

	for(string::const_iterator pos=input.begin(); pos!=input.end(); pos++)
	{
		switch (*pos)
		{
			case 'T': // tree
				binary += "10";
				break;
			case 'F': // food
				binary += "11";
				break;
			case '.': // free 
				binary += "00";
				break;
			default:
				xcs_utility::error(class_name(),"class constructor", "unrecognized map symbol", 1);
		}
	}
}

void
woods_env::reset_input()
{
	path = "";

	current_configuration = 0;
	current_pos_x = free_pos_x[current_configuration];
	current_pos_y = free_pos_y[current_configuration];
	current_position = free_positions[current_configuration];
	set_state();
	
	ostringstream PATH;
	PATH << "(" << current_pos_x << "," << current_pos_y << ")";
	path = PATH.str();
}

bool
woods_env::next_input()
{
	current_configuration++;
	if (current_configuration==no_free_positions)
	{
		reset_problem();
		return false;
	}

	current_pos_x = free_pos_x[current_configuration];
	current_pos_y = free_pos_y[current_configuration];
	current_position = free_positions[current_configuration];

	set_state();
	ostringstream PATH;
	PATH << "(" << current_pos_x << "," << current_pos_y << ")";
	path = PATH.str();
	return true;
}