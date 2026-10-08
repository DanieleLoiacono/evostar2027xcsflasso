/*!
 * \file xcs_utility.cpp
 *
 * \brief implements various utility functions for XCS implementation
 *
 */

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <random>
#include "xcs_random.h"
#include "xcs_utility.h"

using namespace std;

unsigned long 
xcs_utility::binary2long(const string &binary)
{
	unsigned long		power = 1;
	unsigned long		integer = 0;
	string::size_type	bit;

	for(bit=0; bit<binary.size(); bit++)
	{
		integer += (binary[binary.size()-bit-1]-'0') * power;
		power *= 2;
	}
	return integer;
}

string 
xcs_utility::long2binary(const unsigned long decimal, unsigned long size)
{
	string::size_type	bit;
	string			binary;
	unsigned long		base = 1;

	base <<= size;
	for (bit=0; bit<size; bit++)
	{
		base >>= 1;
		binary += '0'+((decimal & base)>0);
	};
	return binary;
}

void 
xcs_utility::set_flag(string set, bool& variable)
{
	if ((set=="on")||(set=="ON"))
		variable = true;
	else if ((set=="off")||(set=="OFF"))
		variable = false;
	else {
		xcs_utility::error("xcs_utility", "set_flag", "value of \'"+set+"\' must be 'on' or 'off'", 1);
	}
}

bool
xcs_utility::string2flag(const string str)
{
	// check for a valid value
	if (str!="on" && str!="off")
		xcs_utility::error("xcs_utility", "string2flag", "value must be 'on' or 'off'", 1);

	if (str=="on")
		return true;
	else
		return false;	
}

bool
xcs_utility::string_to_boolean_option(const string str)
{
	// check for a valid value
	if (str!="on" && str!="off" && str!="yes" && str!="no")
		xcs_utility::error("xcs_utility", "string2flag", "value must be 'on'/'yes' or 'off'/'no'", 1);

	if (str=="on" || str=="yes")
		return true;
	else
		return false;	
}


string
xcs_utility::flag2string(bool flag)
{
	if (flag) 
		return "on";
	else 
		return "off";
}

void 
xcs_utility::error(string name, string method, string message, unsigned int exit_code=1)
{
	cerr << "\n\a***\tERROR";
	cerr <<   "\n***\tCLASS:  \t" << name;
	cerr <<   "\n***\tMETHOD: \t" << method;
	cerr <<   "\n***\tMESSAGE:\t" << message << "." << endl << endl;

	if (exit_code>0)
	{
		exit(exit_code);
	}
}

void 
xcs_utility::warning(string name, string method, string message)
{
	cerr << "\n\a***\tWARNING";
	cerr <<   "\n***\tCLASS:  \t" << name;
	cerr <<   "\n***\tMETHOD: \t" << method;
	cerr <<   "\n***\tMESSAGE:\t" << message << "." << endl << endl;
}

// void 
// xcs_utility::error(string name, string method, t_error_code message, unsigned int exit_code)
// {
// 	cerr << "\n\a***\tERROR    \t";
// 	cerr <<   "\n***\tCLASS:   \t" << name;
// 	cerr <<   "\n***\tMETHOD:  \t" << method;
// 	cerr <<   "\n***\tMESSAGE: \t" << xcs_utility::error_message[message] << "." << endl << endl;
// 	if (exit_code>0)
// 	{
// 		exit(exit_code);
// 	}
// }

string 
xcs_utility::number2string(unsigned long n, unsigned long sz)
{
	ostringstream RESULT;
	RESULT << n;

	string result = RESULT.str();

	for(unsigned long dgt=result.size(); dgt<sz; dgt++)
	{
		result = '0' + result;
	}
	return result;
}

//! delete leading and trailing spaces
string
xcs_utility::trim(string const& source, char const* delims)
{
	string result(source);
	string::size_type index = result.find_last_not_of(delims);

	if(index != std::string::npos)
	{
		result.erase(++index);
	}

	index = result.find_first_not_of(delims);
	if(index != std::string::npos)
		result.erase(0, index);
	else
		result.erase();
	return result;
}

std::vector<std::string> xcs_utility::split(std::string s, std::string delimiter) {
    size_t position_start = 0, position_end, delimiter_length = delimiter.length();
    std::string token;
    std::vector<std::string> result;

    while ((position_end = s.find(delimiter, position_start)) != std::string::npos) {
        token = s.substr (position_start, position_end - position_start);
        position_start = position_end + delimiter_length;
        result.push_back (token);
    }

    result.push_back (s.substr (position_start));
    return result;
}

string 
xcs_utility::remove_comment(string const& source, string comment)
{
	string result(source);
	string::size_type index = result.find(comment);

	return result.substr(0,index);
}

string 
xcs_utility::datetime()
{
	time_t timestamp = time(NULL);
	struct tm datetime = *localtime(&timestamp);

	char output[50];
	strftime(output, 50, "%Y%m%d-%I:%M:%S", &datetime);

	return string(output);	
}