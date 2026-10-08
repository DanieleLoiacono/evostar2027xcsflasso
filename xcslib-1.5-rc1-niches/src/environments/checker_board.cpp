#include "checker_board.h"

bool            checker_board::init;					//!< true if the class has been inited through the configuration manager
unsigned long	checker_board::no_inputs;				//!< number of inputs
double			checker_board::sampling_resolution;	    //!< specify the resolution used to test the learned function
unsigned long	checker_board::no_divisions;			//!< number of divisions on each axes
double			checker_board::tile_size;	            //!< the size of the tile in the board

const std::vector<std::string> checker_board::configuration_parameters = {"number of inputs", "number of divisions","sampling resolution"};

checker_board::checker_board(xcslib::configuration_manager& configuration)
{
	if (!checker_board::init)
	{
		if (!configuration.exist(tag_name()))
		{
			xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
		}

		configuration.check_parameters(tag_name(),checker_board::configuration_parameters);

        try {
            //! function to be approximated
            no_divisions = (unsigned long) configuration.Value(tag_name(), "number of divisions");
        } catch (...) {
            xcs_utility::error(class_name(), "constructor", "attribute \'number of divisions\' not found in <" + tag_name() + ">", 1);
        }

        try {
            //! function to be approximated
            no_inputs = (unsigned long) configuration.Value(tag_name(), "number of inputs");
        } catch (...) {
            xcs_utility::error(class_name(), "constructor", "attribute \'number of inputs\' not found in <" + tag_name() + ">", 1);
        }

        sampling_resolution = configuration.Value(tag_name(), "sampling resolution", 0.05);

		print_parameters(clog);

		current_inputs = vector<double>(no_inputs,0);

		checker_board::init = true;

        tile_size = 1/double(no_divisions);
	}
}

void 
checker_board::print_parameters(ostream &OUTPUT) 
const
{
	OUTPUT << "<" << tag_name() << ">" << endl;
	OUTPUT << "\t" << "number of inputs" << " = " << no_inputs << endl;
	OUTPUT << "\t" << "number of divisions" << " = " << no_divisions << endl;
	OUTPUT << "\t" << "sampling resolution" << " = " << std::fixed << std::setprecision(2) << sampling_resolution << endl;
	OUTPUT << "</" << tag_name() << ">" << endl;
}


void	
checker_board::perform(const t_action& action)
{
    int xi = (int) (current_inputs[0]/tile_size);
    int yi = (int) (current_inputs[1]/tile_size);

    if (action.value()==((xi+yi)%2))
    {
        current_reward = 1000;
    } else {
        current_reward = 0;
    }
}

void
checker_board::begin_problem()
{
	for (int i=0;i<no_inputs;i++)
	{
		current_inputs[i] = xcs_random::random();
	}

    // int xi = (int) (current_inputs[0]/tile_size);
    // int yi = (int) (current_inputs[1]/tile_size);

    // true_action = t_action((xi+yi)%2);
	inputs = real_inputs(current_inputs);
}


bool 
checker_board::allow_test()
const
{
    if (pow(double(no_divisions),double(no_inputs))<1000)
        return true;
    else
        return false;
}

void
checker_board::reset_input()
{
    evaluation_inputs = vector<double>(no_inputs,min_input);

	// for (int i=0;i<no_inputs;i++)
	// {
	// 	evaluation_inputs[i] = min_input;
	// }

    current_inputs = evaluation_inputs;
	inputs = real_inputs(evaluation_inputs);
}

bool 
checker_board::next_input()
{
    bool stop=false;
	evaluation_inputs[no_inputs-1]+=sampling_resolution;

	for(int i=0;i<=no_inputs;i++)
	{
		if (stop == false)
		{
			if (evaluation_inputs[no_inputs-1-i]>=max_input)
			{
				if (no_inputs-1-i == 0)
				{
					return false;
				}
				evaluation_inputs[no_inputs-1-i]=min_input;
				evaluation_inputs[no_inputs-1-i-1]+=sampling_resolution;
			}
			else
			{
				stop = true;
			}
		} else {
            inputs = real_inputs(evaluation_inputs);
            current_inputs = evaluation_inputs;
			return true;
		}
	}

	return false;
}

