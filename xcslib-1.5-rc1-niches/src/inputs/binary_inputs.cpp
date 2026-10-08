#include <string>
#include "binary_inputs.h"

void 
binary_inputs::set_string_value(const string &str)
{
#ifdef __XCS_DEBUG__
	string::size_type	bit;

	bit = 0;
	
	while ((bit<str.size()) && ((str[bit]=='0') || (str[bit]=='1')))
	{
		bit++;
	}

	if (bit!=str.size())
	{	
		xcs_utility::error(class_name(),"set_string_value", "wrong size in assignement.", 1);
	}
#endif
	value = str;
};

// char 
// binary_inputs::input(unsigned long position)
// const
// {
// 	assert(position<size());
// 	return value[position];
// }

// void
// binary_inputs::set_input(unsigned long position, char character)
// {
// 	assert(position<size());
// 	assert( (character=='0') || (character=='1') );
// 	value[position] = character;
// }

void
binary_inputs::numeric_representation(vector<long>& nr)
const
{
	string::const_iterator ch;

	nr.clear();

	for(ch=value.begin(); ch!=value.end(); ch++)
	{
		nr.push_back(long(*ch-'0'));
	}
}

void
binary_inputs::set_numeric_representation(const vector<long>& nr)
{
	vector<long>::const_iterator	val;
	value = "";

	for(val=nr.begin(); val!=nr.end(); val++)
	{
		assert((*val==1)&&(*val==0));
		value += char(*val+'0');
	}
}

void
binary_inputs::numeric_representation(vector<double>& numbers)
const
{
#ifndef __ZERO_AS_NEGATIVE__
	numbers.clear();

	for(string::const_iterator ch=value.begin(); ch!=value.end(); ch++)
	{
		numbers.push_back(double(*ch-'0'));
	}
	
#else
#define V 1

  string::const_iterator ch;

  nr.clear();

  for(ch=value.begin(); ch!=value.end(); ch++)
  {
	//double VV = 1/double(value.size());
	double VV = 5; 
        if (*ch=='0')
          nr.push_back(-VV);
        else
          nr.push_back(VV);
  }

#endif
}

void
binary_inputs::set_numeric_representation(const vector<double>& nr)
{
	vector<double>::const_iterator	val;
	value = "";

	for(val=nr.begin(); val!=nr.end(); val++)
	{
		assert((*val==1)&&(*val==0));
		value += char(*val+'0');
	}
}
