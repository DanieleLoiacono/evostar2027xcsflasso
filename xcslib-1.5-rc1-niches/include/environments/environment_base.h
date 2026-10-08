/*!
 * \class environment_base
 * \brief base class that specifies the basic methods required to implement an environment
 */

#ifndef __ENV_BASE_HH__
#define __ENV_BASE_HH__

using namespace std;

template <typename TInputs, typename TAction>
class environment_base
{
public:
	/*!
	 * \fn string class_name() const
	 * \brief name of the class that implements the environment
	 * 
	 * This function returns the name of the class. 
	 * Every class must implement this method that is used to 
	 * trace errors.
	 */
	//! name of the class that implements the environment
	virtual string class_name() const = 0;
	
	//! tag used to access the configuration file
	virtual string tag_name() const = 0;

	//! defines what has to be done when a new experiment begins
	virtual void begin_experiment() = 0;

	//! defines what has to be done when the current experiment begins
	virtual void end_experiment() = 0;

	//! defines what has to be done when a new problem begins. 

	/*! 
	 * \param explore is true if the problem is done in exploration;
	 * it is false if the problem is solved in exploitation.
	 */
	virtual void begin_problem() = 0;

	//! defines what must be done when the current problem ends
	virtual void end_problem() = 0;

	//! performs action act in the environment 
	virtual void perform(const TAction& act) = 0;

	//! returns the current reward 
	virtual double reward() const = 0;

	//! returns the current state of the environment 
	virtual TInputs state() const = 0;

	//! print the current state of the environment to an output stream. 
	/*!
	 * It is used as a way to pretty print the information about the environmental state;
	 * by default it output the result of the state function, but it can be overriden to 
	 * have richer output.
	 */
	virtual void print(ostream& output) const { output << state();};

	//! returns true if the problem has ended; it is used in multistep problems while it is always true in single step problems
	virtual bool stop() const = 0;
	
	//! writes trace information on an output stream; it is called just before the end_problem method \sa end_problem
	virtual string trace() const = 0;

	//! restore the state of the environment from an input stream
	virtual void restore_state(istream& input) = 0;

	//! save the state of the environment to an output stream
	virtual void save_state(ostream& output) const = 0;

	//! true if the state is terminal and episode must end
	virtual bool is_terminal(const TInputs &state) const = 0;


	//! returns true if the environment implements a single-step problem; false otherwise.
	/*!
	 *  the method can used by the experiment manager.
	 */	
	virtual bool is_single_step() const = 0;
	bool is_multi_step() const { return !is_single_step(); }

	/*! \var unsigned long no_configurations
	 * \brief number of possible input configurations
	 */
	unsigned long			no_configurations;
	unsigned long			current_configuration;	//!<	current visited configuration 
	unsigned long			current_state;			//!<	current visited state
	
	virtual ~environment_base() {};

	//! methods to test environment
	//! true if the environment allows the test of all the possible configurations
	virtual bool allow_test() const {return false;}

	//! reset the environment on the first possible state
	virtual void reset_input() = 0;

	//! returns the next available environmental state; 
	virtual bool next_input() = 0;

	/*! reset the environment on the first possible problem configuration.
	 *  By default, the initial problem states coincides with the feasible inputs 
	 *  we might have cases in which problems can begin only from a subset of feasible inputs.
	 */
	
	virtual void reset_problem() { reset_input(); };

	//! returns the next available problem
	virtual bool next_problem() { return next_input(); };
};
#endif
