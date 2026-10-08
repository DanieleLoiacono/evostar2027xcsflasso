#include <fstream>
#include <iostream>

#include "configuration_manager.h"

xcslib::configuration_manager::configuration_manager(std::string const& extension) 
{
	//! save file extension;
	file_extension = extension;

	//! open the configuration file
	string configFile = "confsys." + extension;
	std::ifstream file(configFile.c_str());

	if (!file.good())
	{
		string msg = "configuration file <" + configFile + "> not opened";
		xcs_utility::error(class_name(),"scan", msg, 1);
	}

#ifdef __DEBUG__
	std::cout << "OPENED <" << configFile << endl;
#endif
	
	//! clear the vector of tags
	tags.clear();
	
  	std::string line;
	std::string input_line;
	
  	std::string name;
  	std::string value;
  	std::string current_section;
  	std::string section_end;
  	
	int posEqual;
  	while (std::getline(file,input_line)) 
	{
		//! eliminate comments
		line = xcs_utility::remove_comment(input_line);

		//! clear the line from leading and trailing spaces
		line=xcs_utility::trim(line);
		
    		//! does not consider empty lines
    		if (!line.length()) continue;

		// //! does not consider comments
		// if (line[0] == '#') continue;
		// if (line[0] == ';') continue;
	
		//! check for the start/end of a section
		if (input_line[0] == '<') 
		{

			//! check if is an end
			if (input_line[1]=='/')
			{
				//! end of a section
				section_end=xcs_utility::trim(line.substr(2,line.find('>')-2));

				//cout << "*** " << current_section << endl;
				//cout << "*** " << section_end << endl;

				if (section_end!=current_section)
				{
					string msg = "section <"+current_section+"> ends with </"+section_end+">";
					xcs_utility::error(class_name(),"constructor", msg, 1);
				} else {
					//! clear current section
					current_section = "";
				}

			} else {

				//! start of a new section
				if (current_section!="")
				{
					xcs_utility::error(class_name(),"constructor", "nested section in <"+current_section+">", 1);
				}

				//! begin of a section
				current_section=xcs_utility::trim(line.substr(1,line.find('>')-1));
				tags.push_back(current_section);

				//! check for '/' symbols in tags
				if (current_section.find('/')!=std::string::npos)
				{
					string msg = "symbol '/' not allowed in tag " + section_end;
					xcs_utility::error(class_name(),"constructor", msg, 1);
				}
			}

#ifdef __DEBUG__
			std::cout << "TAG <" << inSection << ">" << endl;
#endif
			continue;
		}

		posEqual=line.find('=');
		
		if (posEqual==std::string::npos)
		{
			string msg = "configuration entry <" + line + "> not a comment, not a tag, not an assignment";
			xcs_utility::error(class_name(),"constructor", msg, 1);
		}
			
		name  = xcs_utility::trim(line.substr(0,posEqual));
		value = xcs_utility::trim(line.substr(posEqual+1));
			
#ifdef __DEBUG__
		std::cout << "\tATTRIBUTE <" << name << ">" << endl;
		std::cout << "\tVALUE     <" << value << ">" << endl;
#endif
		content2_.push_back(pair<string,xcslib::generic>(current_section+'/'+name, xcslib::generic(value)));
		content_[current_section+'/'+name]=xcslib::generic(value);
	}
}

void 
xcslib::configuration_manager::check_parameters(std::string const& section, const std::vector<string>& configuration_parameters)
{
	// std::map<std::string, xcslib::generic>::iterator it
	for (auto it = content_.begin(); it!=content_.end(); it++)
	{
		std::string str_key = it->first;
		
		int slash_position = str_key.find('/');

		if (slash_position==std::string::npos)
		{
			xcs_utility::error(class_name(),"check_parameters", "something went wrong reading the configuration", 1);
		}

		std::string str_section = str_key.substr(0,slash_position);

		if (str_section==section)
		{
			std::string str_parameter = str_key.substr(slash_position+1);

			vector<string>::const_iterator parameter_position = find(configuration_parameters.begin(),configuration_parameters.end(),str_parameter);

			if (parameter_position==configuration_parameters.end())
			{
				xcs_utility::error(class_name(),"check_parameters", "parameter \'"+str_parameter+"\' in section " + section + " is not supported", 1);
			}
		}

	}
}
void
xcslib::configuration_manager::save(ostream &output) const
{
	for(vector<string>::const_iterator tag = tags.begin(); tag!=tags.end(); tag++)
	{
		string current_tag = *tag;
		
		output << "<" << current_tag << ">" << endl;
		
		for(vector< pair<string,xcslib::generic> >::const_iterator ci=content2_.begin(); ci!=content2_.end(); ci++)
		{
			string s = (ci->first);				
			
			string t = s.substr(0,s.find('/'));	//! tag in list
			string p = s.substr(s.find('/')+1);		//! associated generic

			/*output << "*** " << s << endl;
			output << "*** " << t << endl;
			output << "*** " << p << endl;*/

			xcslib::generic v = ci->second;			//! associated value
			
			if (current_tag == t)
			{
				output << "\t" << p << " = " << v << endl;
			}
			
		}
		
		output << "</" << current_tag << ">" << endl;
	}
}

bool 
xcslib::configuration_manager::section_contains_entry(std::string const& section, std::string const& entry)
const
{
	std::map<std::string,xcslib::generic>::const_iterator ci = content_.find(section + '/' + entry);

	return ci!=content_.end();
}


xcslib::generic const& xcslib::configuration_manager::Value(std::string const& section, std::string const& entry) const
{
#ifdef __DEBUG__
	for(std::map<std::string,xcslib::generic>::const_iterator ci=content_.begin(); ci!=content_.end(); ci++)
	{
		cout << ci->first << endl;
	}
#endif

	std::map<std::string,xcslib::generic>::const_iterator ci = content_.find(section + '/' + entry);

	if (ci == content_.end()) throw string(entry).c_str();
	return ci->second;
}

xcslib::generic const& xcslib::configuration_manager::Value(std::string const& section, std::string const& entry, double value)
{
	try {
		return Value(section, entry);
	} catch(const char*) {
		return content_.insert(std::make_pair(section+'/'+entry, xcslib::generic(value))).first->second;
	}
}

xcslib::generic const& xcslib::configuration_manager::Value(std::string const& section, std::string const& entry, unsigned long value)
{
	try {
		return Value(section, entry);
	} catch(const char*) {
		return content_.insert(std::make_pair(section+'/'+entry, xcslib::generic(value))).first->second;
	}
}

xcslib::generic const& xcslib::configuration_manager::Value(std::string const& section, std::string const& entry, std::string const& value)
{
	try {
		return Value(section, entry);
	} catch(const char*) {
		return content_.insert(std::make_pair(section+'/'+entry, xcslib::generic(value))).first->second;
	}
}
