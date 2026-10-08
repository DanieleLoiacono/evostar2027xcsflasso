#include <deque>
#include "xcs_definitions.h"

#include "pf/base.h"
#include "pf/utility.h"

#ifndef __xcsf_classifier__
#define __xcsf_classifier__


/*!
 * \class xcsf_classifier
 *
 * \brief class definition for XCS classifiers
 */
class xcsf_classifier
{
 public:
	//! name of the class that implements XCS classifiers
	/*!
	 * \fn string class_name() 
	 * \brief name of the class that implements the environment
	 * 
	 * This function returns the name of the class. 
	 * Every class must implement this method that is used to 
	 * trace errors.
	 */
	string class_name() const { return string("xcsf_classifier"); };

	//! class tag used in the configuration file; currently not used 
	string tag_name() const { return string("classifier"); };

 public:
	//! class constructor
	xcsf_classifier();

	//! class constructor
	xcsf_classifier(const xcsf_classifier&);

	//! class destructor
	~xcsf_classifier()
	{
		delete prediction_function;
		prediction_function = NULL;		
	};

	friend bool operator==(const xcsf_classifier&, const xcsf_classifier&);
	friend bool operator<(const xcsf_classifier&, const xcsf_classifier&);
	friend bool operator!=(const xcsf_classifier&, const xcsf_classifier&);

	//! assignment operator for a constant value
	xcsf_classifier& operator=(const xcsf_classifier&);

	//! return the label associated with the element in position fld returned by the stream operator
	string	stream_field(unsigned long fld) {return "";};

	//! write the classifier to an output stream
	friend ostream& operator <<(ostream&, const xcsf_classifier&);

	//! read the classifier from an input stream
	friend istream& operator >>(istream&, xcsf_classifier&);

	//! return the label associated with the element in position fld returned by the stream operator
	string	print_field(unsigned long fld) const { return ""; };

	//! save the state of the classifier class to an output stream
	static void	save_state(ostream& output) { output << id_count << endl;};

	//! restore the state of the classifier class from an input stream
	static void	restore_state(istream& input) {input >> id_count;};

	//! generate a random classifier
	void	random();			

	//! cover the current input
	void	cover(const t_state&);

	//! match the current input
	bool	match(const t_state&);

	//! mutate the classifier according to the mutation probability "mu"
	void	mutate(const float, const t_state&); 

	//! apply crossover between this classifier and another one
	void	recombine(xcsf_classifier& classifier);
	
	//! return true if this classifier subsumes the classifier "cs"
	bool	subsume(const xcsf_classifier& classifier) const;

	//! return the classifier id
	unsigned long	id() const {return identifier;};		
	
	//! generate the unique identifier (used when inserting the classifier in the population)
	void generate_id() {identifier = ++xcsf_classifier::id_count;};	// 

	//! it should be resetted every experiment
	static void reset_id() {xcsf_classifier::id_count = 0;}

	//! 
	void read_from_string(string line);

	//! computed prediction

	//! return computed classifier prediction
	double get_prediction(vector <double> input ){return prediction_function->output(input);};
	
	//! update classifier prediction parameters
	void update_prediction(vector <double> input, double target, double gradient_term){prediction_function->update(input,target,gradient_term);};
	void update_prediction(vector <double> input, double target){prediction_function->update(input,target);};


 private:
	//!  set the classifier parameters to default values
	inline void set_initial_values();	

 public:
	bool is_more_general_than(const xcsf_classifier &classifier) const { return condition.is_more_general_than(classifier.condition); };

 private:
	static 	unsigned long	id_count;		//!< global counter used to generate classifier identifiers
	static	unsigned int	output_precision;	//!< precision of the output (useless)

 public:
	unsigned long		identifier;		//!< classifier identifier; it is unique for each classifier

	t_condition			condition;		//!< classifier condition
	t_action			action;			//!< classifier action

	double				prediction;		// //!< prediction (not used but kept for compatibility with code shared with XCS)
	double				error;			// //!< prediction error
	double				fitness;		// //!< classifier fitness
	double				actionset_size;	// //!< estimate of the size of the action set [A]
	
	unsigned long		experience;		// //!< classifier experience, i.e., the number of times that the classifier has een updated
	unsigned long		numerosity;		// //!< classifier numerosity, i.e., the number of micro classifiers
	unsigned long		time_stamp;		// //!< time of the last genetic algorithm application

	//! XCSF specific parameters
	xcsflib::base_pf *prediction_function;	//!< prediction function
	double qerror;				//!< prediction quadratic error


#ifdef __NICHE_TRACKING__
//! Niche tracking
public:
	unsigned long		creation_time_stamp;		// //!< time when the classifier was created
	unsigned long		as_time_stamp;				// //!< last time the classifier was in the action set size

	void update_time_stamps(unsigned long time_stamp, unsigned long size = 20);
	void save_time_stamps(ostream& output) const;
	deque<unsigned long> load_time_stamps(string str_time_stamps, string delimiter="|");
	void init_time_stamps() {as_time_stamps.clear();}

private:
	deque<unsigned long> as_time_stamps; 			// //!< maintains the time stamps of all the action sets
#endif

};
#endif
