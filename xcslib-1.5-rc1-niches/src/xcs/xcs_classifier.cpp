#include <ostream>
#include <sstream>

#include "xcs_classifier.h"
#include "xcs_utility.h"

using namespace std;

unsigned long		xcs_classifier::id_count = 0;
unsigned int		xcs_classifier::output_precision = 5;	


//! class constructor
xcs_classifier::xcs_classifier()
{
	identifier = xcs_classifier::id_count++;
	numerosity = 1;
	time_stamp=0;
	experience=0;
	prediction=0;
	error=0;
	fitness=0;
	actionset_size=0;
	creation_time_stamp = 0;			//!< time stamp when the classifier was created

#ifdef __NICHE_TRACKING__
	as_time_stamp = 0;					//!< last time when the classifier was in the action set
	as_time_stamps.clear();
#endif
}

//! copy constructor
xcs_classifier::xcs_classifier(const xcs_classifier& classifier)
{
	identifier = classifier.identifier;
	condition = classifier.condition;
	action = classifier.action;
	prediction = classifier.prediction;		//!< prediction 
	error = classifier.error;			//!< prediction error
	fitness = classifier.fitness;			//!< classifier fitness
	actionset_size = classifier.actionset_size;	//!< estimate of the size of the action set [A]
	
	experience = classifier.experience;		//!< classifier experience, i.e., the number of times that the classifier has een updated
	numerosity = classifier.numerosity;		//!< classifier numerosity, i.e., the number of micro classifiers
	time_stamp = classifier.time_stamp;		//!< time of the last genetic algorithm application
	creation_time_stamp = classifier.creation_time_stamp;		//!< time stamp when the classifier was created

#ifdef __NICHE_TRACKING__
	as_time_stamp = classifier.as_time_stamp;					//!< last time when the classifier was in the action set
	as_time_stamps = classifier.as_time_stamps;					//!< copies the list of [A] identifiers
#endif
}

//! comparison operators 

bool 
operator==(const xcs_classifier& classifier1, const xcs_classifier& classifier2)
{
	return ((classifier1.condition == classifier2.condition) && (classifier1.action == classifier2.action));
}

bool 
operator!=(const xcs_classifier& classifier1, const xcs_classifier& classifier2)
{
	return !(classifier1==classifier2);
}

bool 
operator<(const xcs_classifier& classifier1, const xcs_classifier& classifier2)
{
	return ( (classifier1.condition < classifier2.condition) ||  
		 ((classifier1.condition == classifier2.condition) && (classifier1.action < classifier2.action)) );
}

//! stream operators

ostream&
operator<<(ostream& output,const xcs_classifier& classifier)
{
	output << classifier.id() << '\t';
	output << classifier.condition << "\t";
	output << classifier.action << '\t';
	output.setf(ios::scientific);
	output.precision(xcs_classifier::output_precision);
	output << classifier.prediction << '\t';
  	output.precision(xcs_classifier::output_precision);
  	output << classifier.error << '\t';
  	output.precision(xcs_classifier::output_precision);
  	output << classifier.fitness << '\t';
  	output.precision(xcs_classifier::output_precision);
  	output << classifier.actionset_size << '\t';
  	output.precision(xcs_classifier::output_precision);
  	output << classifier.experience << '\t';
  	output.precision(xcs_classifier::output_precision);
  	output << classifier.numerosity;

#ifdef __NICHE_TRACKING__
	output << '\t';
  	output << classifier.creation_time_stamp << '\t';
  	output << classifier.time_stamp << '\t';
  	output << classifier.as_time_stamp << '\t';	
	classifier.save_time_stamps(output);
#endif

	return (output);
}


//! read the classifier from an input stream
istream&
operator>>(istream& is, xcs_classifier& classifier)
{
	long double prediction;
	long double error;
	long double fitness;
	long double actionset_size;

	if (!(is>>classifier.identifier))
	{
		xcs_utility::error(classifier.class_name(), ">>", "identifier failed to read", 1);
	} else {
		clog << "identifier " << classifier.identifier << endl;
	}

	if (!(is>>classifier.condition))
	{
		xcs_utility::error(classifier.class_name(), ">>", "condition failed to read", 1);
	}
		
	if (!(is>>classifier.action))
	{
		xcs_utility::error(classifier.class_name(), ">>", "action failed to read", 1);
	}
		
	if (!(is>>prediction))
	{
		xcs_utility::error(classifier.class_name(), ">>", "prediction failed to read", 1);
	} else {
		classifier.prediction = prediction;
	}
		
	if (!(is >> error))
	{
		xcs_utility::error(classifier.class_name(), ">>", "error failed to read", 1);
	} else {
		classifier.error = error;
	}

	if (!(is >> fitness))
	{
		xcs_utility::error(classifier.class_name(), ">>", "fitness failed to read", 1);
	} else {
		classifier.fitness = fitness;
	}

	if (!(is >> actionset_size))
	{
		xcs_utility::error(classifier.class_name(), ">>", "actionset_size failed to read", 1);
	}else {
		classifier.actionset_size = actionset_size;
	}

	if (!(is >> classifier.experience))
	{
		xcs_utility::error(classifier.class_name(), ">>", "experience failed to read", 1);
	}

	if (!(is >> classifier.numerosity))
	{
		xcs_utility::error(classifier.class_name(), ">>", "numerosity failed to read", 1);
	}

#ifdef __NICHE_TRACKING__
	if (!(is >> classifier.creation_time_stamp))
		xcs_utility::error(classifier.class_name(), ">>", "creation_time_stamp failed to read", 1);

	if (!(is >> classifier.time_stamp))
		xcs_utility::error(classifier.class_name(), ">>", "time_stamp failed to read", 1);

	if (!(is >> classifier.as_time_stamp))
		xcs_utility::error(classifier.class_name(), ">>", "as_time_stamp failed to read", 1);

	string str_time_stamps; 
	if (!(is >> str_time_stamps))
		xcs_utility::error(classifier.class_name(), ">>", "time stamp list failed to read", 1);
	else {
		classifier.as_time_stamps = classifier.load_time_stamps(str_time_stamps);
	}
#endif

	return (is);
}

bool	
xcs_classifier::match(const t_state& inputs)
{
	return (condition.match(inputs));
}

void	
xcs_classifier::random()
{
	condition.random();
	action.random();
	set_initial_values();
}

void	
xcs_classifier::cover(const t_state& inputs)
{
	condition.cover(inputs);
	action.random();
	set_initial_values();
}

void
xcs_classifier::mutate(float mutation_probability, const t_state& inputs) 
{
	condition.mutate(mutation_probability,inputs);
	action.mutate(mutation_probability);
}

void
xcs_classifier::recombine(xcs_classifier& classifier)
{
	condition.recombine(classifier.condition);
	swap(action,classifier.action);
}

bool 
xcs_classifier::subsume(const xcs_classifier& classifier)
const
{
	return ((action==classifier.action) && this->condition.is_more_general_than(classifier.condition));
};


inline
void 
xcs_classifier::set_initial_values()
{
	identifier = xcs_classifier::id_count++;
	numerosity = 1;
	time_stamp=0;
	experience=0;
	prediction=0;
	error=0;
	fitness=0;
	actionset_size=0;

#ifdef __NICHE_TRACKING__
	creation_time_stamp = 0;			//!< time stamp when the classifier was created
	as_time_stamp = 0;					//!< last time when the classifier was in the action set
	as_time_stamps.clear();
#endif
}

// void 
// xcs_classifier::read_from_string(const string& line)
// {
// 	std::vector<std::string> fields = xcs_utility::split(line, "\t");
// 	cout << "# fields = " << fields.size() << endl;

// 	identifier = stoul(fields[0]);
// 	// condition = fields[1];
// 	// action = fields[2];
// 	prediction = stod(fields[3]);
// 	error = stod(fields[4]);
// 	fitness = stod(fields[5]);
// }

//! assignment operator for a constant value
xcs_classifier& 
xcs_classifier::operator=(const xcs_classifier& classifier)
{
	set_initial_values();	
	condition = classifier.condition;
	action = classifier.action;
	prediction = classifier.prediction;		//!< prediction 
	error = classifier.error;			//!< prediction error
	fitness = classifier.fitness;			//!< classifier fitness
	actionset_size = classifier.actionset_size;	//!< estimate of the size of the action set [A]
	
	experience = classifier.experience;		//!< classifier experience, i.e., the number of times that the classifier has een updated
	numerosity = classifier.numerosity;		//!< classifier numerosity, i.e., the number of micro classifiers

#ifdef __NICHE_TRACKING__
	time_stamp = classifier.time_stamp;							//!< last time when the genetic algorithm was run
	creation_time_stamp = classifier.creation_time_stamp;		//!< time stamp when the classifier was created
	as_time_stamp = classifier.as_time_stamp;					//!< last time when the classifier was in the action set
	as_time_stamps = classifier.as_time_stamps;					//!< copies the list of [A] identifiers
#endif

	return *this;
}

#ifdef __NICHE_TRACKING__
void 
xcs_classifier::update_time_stamps(unsigned long time_stamp, unsigned long size)
{
	as_time_stamps.push_back(time_stamp);

	if (size==0)
		return;

	if (as_time_stamps.size()>size)
	{
		as_time_stamps.pop_front();
	}
}

void 
xcs_classifier::save_time_stamps(ostream& output) const
{
	ostringstream str;
	str << "[";

	if (as_time_stamps.size()>0)
		str << as_time_stamps[0];

	for (int i=1; i<as_time_stamps.size();i++)
	{
		str << "|"<< as_time_stamps[i];
	}
	str << "]";

	output << str.str();	
}

deque<unsigned long> xcs_classifier::load_time_stamps(string str_time_stamps, string delimiter)
{

	deque<unsigned long> as_time_stamps;

    if (str_time_stamps=="[]")
    	return as_time_stamps;
        
	string str_stamps = str_time_stamps.substr(1,str_time_stamps.size()-2);
	std::vector<std::string> tokens = xcs_utility::split(str_stamps, delimiter);

	for(size_t i=0; i<tokens.size();i++)
    {
    	unsigned long token = atol(tokens[i].c_str());
                
    	as_time_stamps.push_back(token);
    }

    
    return as_time_stamps;
}
#endif
