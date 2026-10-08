#include <deque>
#include "xcs_definitions.h"

#ifndef __XCS_CLASSIFIER__
#define __XCS_CLASSIFIER__


/*!
 * \class xcs_classifier
 *
 * \brief class definition for XCS classifiers
 */
class xcs_classifier
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
	string class_name() const { return string("xcs_classifier"); };

	//! class tag used in the configuration file; currently not used 
	string tag_name() const { return string("classifier"); };

 public:
	//! class constructor
	xcs_classifier();

	//! class constructor
	xcs_classifier(const xcs_classifier&);

 public:

	//! comparison operators
	friend bool operator==(const xcs_classifier&, const xcs_classifier&);
	friend bool operator<(const xcs_classifier&, const xcs_classifier&);
	friend bool operator!=(const xcs_classifier&, const xcs_classifier&);

	//! assignment operator for a constant value
	xcs_classifier& operator=(const xcs_classifier&);

	//! return the label associated with the element in position fld returned by the stream operator
	// string	stream_field(unsigned long fld) {return "";};

	//! write the classifier to an output stream
	friend ostream& operator <<(ostream&, const xcs_classifier&);

	//! read the classifier from an input stream
	friend istream& operator >>(istream&, xcs_classifier&);

	// //! return the label associated with the element in position fld returned by the stream operator
	// string	print_field(unsigned long fld) const { return ""; };

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
	void	mutate(float, const t_state&); 

	//! apply crossover between this classifier and another one
	void	recombine(xcs_classifier& classifier);
	
	//! return true if this classifier subsumes the classifier "cs"
	bool	subsume(const xcs_classifier& classifier) const;

	//! return the classifier id
	unsigned long	id() const {return identifier;};		
	
	//! generate the unique identifier (used when inserting the classifier in the population)
	void generate_identifier() {identifier = ++xcs_classifier::id_count;};	// 

	//! it should be resetted every experiment
	static void reset_id() {xcs_classifier::id_count = 0;}

	//! 
	// void read_from_string(const string& line);

 private:
	//!  set the classifier parameters to default values
	inline void set_initial_values();	

 public:
	bool is_more_general_than(const xcs_classifier &classifier) const { return condition.is_more_general_than(classifier.condition); };

 private:
	static 	unsigned long	id_count;			//!< global counter used to generate classifier identifiers
	static	unsigned int	output_precision;	//!< precision of the output (useless)

 public:
	unsigned long		identifier;		//!< classifier identifier; it is unique for each classifier

	t_condition			condition;		//!< classifier condition
	t_action			action;			//!< classifier action

	double				prediction;		// //!< prediction 
	double				error;			// //!< prediction error
	double				fitness;		// //!< classifier fitness
	double				actionset_size;	// //!< estimate of the size of the action set [A]
	
	unsigned long		experience;		// //!< classifier experience, i.e., the number of times that the classifier has een updated
	unsigned long		numerosity;		// //!< classifier numerosity, i.e., the number of micro classifiers
	unsigned long		time_stamp;		// //!< time of the last genetic algorithm application
	unsigned long		creation_time_stamp;		// //!< time when the classifier was created


#ifdef __NICHE_TRACKING__
//! Niche tracking
public:
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
