/*!
 * \namespace xcs_utility
 *
 * \brief functions which are needed in various parts of XCS code.
 *
 * This namespace collect the common functions used in different part of 
 * XCS code. It includes functions for binary to integer conversion, integer to binary
 * conversion, parameter settings, etc.
 *
 * \author Pier Luca Lanzi
 *
 * \version 0.01
 *
 * \date 2002/05/28
 *
 */

#include <sys/times.h>
#include <ctime>
#include <iomanip>
#include <unistd.h>
#include <string>
#include <vector>
#include <algorithm>
#include <ctime>

using namespace std;

#ifndef XCS_UTILITY_H
#define XCS_UTILITY_H

namespace xcs_utility {

	//! convert a string that represents a binary number to a unsigned long
	unsigned long binary2long(const string&);

	//! convert an unsigned long to a binary string
	string long2binary(const unsigned long, unsigned long);

	//! conver a number to string filling up sz positions;
	string number2string(unsigned long n, unsigned long sz);

	//! set the Boolean variable to true if the string is "on"; to false if the string is "off"
	void set_flag(string set, bool& var);

	bool string_to_boolean_option(const string str);

	bool string2flag(const string str);
	string flag2string(bool f);

	//! print an error message specified with a string 
	void error(string name, string method, string message, unsigned int exit_code);
	
	//! print a warning message specified with a string 
	void warning(string name, string method, string message);
	
	// //! print an error message specified with a a code \sa t_error_code
	// void error(string name, string method, t_error_code message, unsigned int exit_code);

	string trim(string const& source, char const* delims = " \t\r\n");
	
	string remove_comment(string const& source, string comment = "//");

	vector<string> split(std::string s, std::string delimiter);

	string datetime();
};


#endif
