/* 
   generic.h

   Derived from the Chamaleon class 

   Copyright (C) 2002-2004 RenM-CM-) Nyffenegger

   This source code is provided 'as-is', without any express or implied
   warranty. In no event will the author be held liable for any damages
   arising from the use of this software.

   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:

   1. The origin of this source code must not be misrepresented; you must not
      claim that you wrote the original source code. If you use this source code
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original source code.

   3. This notice may not be removed or altered from any source distribution.

   RenM-CM-) Nyffenegger rene.nyffenegger@adp-gmbh.ch
*/

#ifndef __GENERIC_HPP__
#define __GENERIC_HPP__

#include <string>
#include <iostream>

namespace xcslib
{

	class generic 
	{
	public:
		generic() {};
			
		explicit generic(const std::string&);
		explicit generic(double);
		explicit generic(long);
		explicit generic(unsigned long);
		explicit generic(const char*);
	
		generic(const generic&);
		generic& operator=(generic const&);
		generic& operator=(double);
		generic& operator=(std::string const&);
		
	public:
		operator std::string	() const;
		operator double		() const;	
		operator unsigned long	() const;
		operator long		() const;
	
		//! write the condition to an output stream 
		friend std::ostream& operator<<(std::ostream& output, const generic& ch)
		{
			output << ch.value_;
			return (output);
		};
	
		//! read the condition from an input stream 
		friend std::istream& operator>>(std::istream& input, generic& ch)
		{
			std::string str;
			input >> str;
			ch.value_ = str; 
			return (input);
		};
	
	private:
		std::string value_;
	};

}
#endif
