#ifndef __XCS_CONFIG_MGR2__
#define __XCS_CONFIG_MGR2__

#include <string>
#include <map>

#include "generic.h"
#include "xcs_utility.h"

namespace xcslib
{
	class configuration_manager {

		//! values in the configuration file
		std::map<std::string, xcslib::generic> 	content_;
		std::vector< pair<string,xcslib::generic> >	content2_;
		
		//! index of all the tags in the configuration file
		std::vector<std::string>		tags;
		
		//! extension of the configuration file
		std::string				file_extension;

	public:	

		//! name of the class
		std::string class_name() const { return std::string("xcs_configuration_manager"); };

		//! Constructor for the configuration manager class 
		/*!
		*  This is the only constructor.
		*/
		configuration_manager(std::string const& extension);

		//! return the file extension that is used by the configuration manager
		string extension() const {return file_extension;};

		//! true if a section labeled tag has been found in the configuration file 
		bool exist(const std::string &tag) const {return find(tags.begin(), tags.end(), tag)!=tags.end(); };

		//! true if an entry exists in a section
		bool section_contains_entry(std::string const& section, std::string const& entry) const;
		
		//! save the current configuration to an output stream 
		void save(ostream&) const;


		
		xcslib::generic const& Value(std::string const& section, std::string const& entry) const;
		xcslib::generic const& Value(std::string const& section, std::string const& entry, double value);
		xcslib::generic const& Value(std::string const& section, std::string const& entry, unsigned long value);
		xcslib::generic const& Value(std::string const& section, std::string const& entry, std::string const& value);

		void check_parameters(std::string const& section, const std::vector<string>& configuration_parameters);
	};

}

#endif
