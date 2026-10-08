#ifndef __XCSLIB_COORDINATE__
#define __XCSLIB_COORDINATE__

#include <string>
#include <sstream>

namespace xcslib {
    class coordinate 
    {
        public:
            double x;
            double y;

            double get_x() const { return x;}
            double get_y() const { return y;}

            coordinate(): x(0), y(0) {};
            coordinate(const double x0, const double y0): x(x0),y(y0) {};

            std::string string_value() const {	
                std::ostringstream sstr;
                sstr << x << "," << y;
                return sstr.str();
            };

            friend double squared_distance(const coordinate& a, const coordinate &b)
            {
                return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
            };

            //! euclidean distance
            friend double euclidean_distance(const coordinate &a, const coordinate &b)
            {
                return sqrt(squared_distance(a, b));
            }

            // Function to find orientation of the triplet (a, b, c)
            // Returns -1 if clockwise, 1 if counter-clockwise, 0 if collinear
            friend int orientation(xcslib::coordinate a, xcslib::coordinate b, xcslib::coordinate c) 
            {
                double v = a.x * (b.y - c.y) +
                        b.x * (c.y - a.y) +
                        c.x * (a.y - b.y);
                if (v < 0) return -1;
                if (v > 0) return +1;
                return 0;
            }            

            bool operator==(const coordinate& pt) const { return ((x==pt.x) && (y==pt.y)); };
            bool operator!=(const coordinate& pt) const { return ((x!=pt.x) || (y!=pt.y)); };
            bool operator<(const coordinate& pt)  const { if (x<pt.x) return true; else return ((x==pt.x) && (y<=pt.y)); };
    };
}

#endif
