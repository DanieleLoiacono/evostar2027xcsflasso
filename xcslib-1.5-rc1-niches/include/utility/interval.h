#define __INTERVAL_TEXT_FORMAT__ "[%lf;%lf]"

/*!
 * \file interval.h
 *
 * \author Pier Luca Lanzi
 *
 * \version 1.0
 *
 * \date
 *
 * \brief generic intervals
 *
 */

/*!
 * \class xcslib::interval<T> interval.h
 *
 * \brief definition class for generic intervals
 */

#ifndef __GENERIC_INTERVAL__
#define __GENERIC_INTERVAL__

#include <string>
#include <sstream>

using namespace std;

namespace xcslib {
	
	template<typename T>
	class interval
	{
	private:
		T lower_bound;
		T upper_bound;
	public:
		//! constructors
		
		//! empty constructor
		interval () {};
	
		//!
		interval (T lower, T upper)
		{
			lower_bound = lower;
			upper_bound = upper;				
			assert(lower_bound <= upper_bound);
		};
	
		//! copy constructor
		interval (const interval<T>& i)
		{
			lower_bound = i.lower_bound;
			upper_bound = i.upper_bound;
		};
	
		//! destructor
		~interval<T> (){};
	
		//! set lower bound
		void set_lower_bound (T lower) 
		{
			lower_bound = lower;
			if (lower_bound > upper_bound)
				upper_bound=lower_bound;
		};
	
		//! set upper bound
		void set_upper_bound (T upper)
		{
			upper_bound = upper;
			if (lower_bound > upper_bound)
				lower_bound = upper_bound;
		};

		bool contains(T value) const
		{
			return (value>lower_bound) && (value<=upper_bound);
		}

		bool contains(T value, T min_value, T max_value) const
		{
			//! either min_value < value <= max_value or the value is actually the lower bound
			bool lower = ((lower_bound==min_value) && (value == min_value)) || (lower_bound<value); 
			bool upper = (value<=upper_bound);
			return lower && upper;
		}
		
		//! set lower bound
		T get_lower_bound () const {return lower_bound;};
	
		//! set upper bound
		T get_upper_bound () const {return upper_bound;};
	
		//! true when the interval is contained in interval i
		bool operator< (const interval<T>& i) const
		{
			return ((lower_bound >= i.get_lower_bound())&&(upper_bound <= i.get_upper_bound()));
		};
	
		//! equality operator
		bool operator==(const interval<T>& i) const
		{
			return ((lower_bound == i.get_lower_bound())&&(upper_bound == i.get_upper_bound()));
		};
	
		//! inequality operator
		bool operator!=(const interval<T>& i) const
		{
			return ((lower_bound != i.get_lower_bound())||(upper_bound != i.get_upper_bound()));
		};
	
		//! assignment operator
		interval<T>& operator=(interval<T>& i)
		{
			lower_bound = i.get_lower_bound();
			upper_bound = i.get_upper_bound();
			return (*this);
		};
	
		//! assignment operator for a constant value
		interval<T>& operator=(const interval<T>& i)
		{
			lower_bound = i.get_lower_bound();
			upper_bound = i.get_upper_bound();
			return (*this);
		};
	
		string string_value()
		const
		{
	#ifndef __INTERVAL_TEXT_FORMAT__
			ostringstream str_int;
	
			str_int << "[";
			str_int << lower_bound;
			str_int << ";";
			str_int << upper_bound;
			str_int << "]";
	
			return str_int.str();
	#else
			char c[500];
			if (!snprintf(c, 500, __INTERVAL_TEXT_FORMAT__,lower_bound,upper_bound))
			{
				xcs_utility::error("real_interval","to_string()","real_interval text format not correct",1);
			}
			return string(c);
	#endif
		}
	
		void set_string_value(string str)
		{
	#ifndef __INTERVAL_TEXT_FORMAT__
			char ls, rs;	//! left and right squared parentheses
			char sep;	//! separator
	
			istringstream str_int(str);
	
			str_int >> ls;
			str_int >> lower_bound;
			str_int >> sep;
			str_int >> upper_bound;
			str_int >> rs;
	
			assert(lower_bound <= upper_bound);
			assert(ls=='[' && rs==']' && sep==';');
	#else
			if(!sscanf(str.c_str(),__INTERVAL_TEXT_FORMAT__,&lower_bound,&upper_bound))
			{
				xcs_utility::error("real_interval","to_string()","real_interval text format not correct",1);
			}
			assert(lower_bound <= upper_bound);
	// 		return (*this);
	#endif
		}
	
		void
		swap(interval<T>& i)
		{
			T tmp_lower = i.get_lower_bound();
			T tmp_upper = i.get_upper_bound();
		
			i.set_lower_bound(lower_bound);
			i.set_upper_bound(upper_bound);
		
			lower_bound = tmp_lower;
			upper_bound = tmp_upper;
		}
	
		void
		swap_lower(interval<T>& i)
		{
			T tmp_lower = i.get_lower_bound();
			i.set_lower_bound(lower_bound);
			lower_bound = tmp_lower;
			if (lower_bound > upper_bound)
				upper_bound = lower_bound;
		}
	
		void
		swap_upper(interval<T>& i)
		{
			T tmp_upper = i.get_upper_bound();
			i.set_upper_bound(upper_bound);
			upper_bound = tmp_upper;
			if (lower_bound > upper_bound)
				lower_bound = upper_bound;
		}
	};

};	//! end namespace xcslib
#endif
