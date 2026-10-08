#include <map>
#include "generic.h" 

namespace xcslib
{
    class statistics
    {
        private:
            std::map<std::string, xcslib::generic> 	content_;

        public:
            xcslib::generic& operator[](const std::string &tag);
        //     xcslib::generic const& set(std::string const& section, std::string const& entry) const;
		//     xcslib::generic const& Value(std::string const& section, std::string const& entry, double value);
		// xcslib::generic const& Value(std::string const& section, std::string const& entry, unsigned long value);
		// xcslib::generic const& Value(std::string const& section, std::string const& entry, std::string const& value);


    }
}