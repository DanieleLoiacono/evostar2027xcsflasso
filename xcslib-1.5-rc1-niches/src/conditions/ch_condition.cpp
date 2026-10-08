#include <cassert>
#include <string>
#include <fstream>
#include <algorithm>
#include "xcs_definitions.h"
#include "xcs_random.h"
#include "configuration_manager.h"
#include "ch_condition.h"

// #include <CGAL/Cartesian.h>
// #include <CGAL/ch_graham_andrew.h>

using namespace std;

// #define __INTERVAL_TEXT_FORMAT__ "[%lf;%lf]"
// #define __INTERVAL_GRAPH_OPEN_CHAR__ '|'
// #define __INTERVAL_GRAPH_CLOSE_CHAR__ '|'
// #define __INTERVAL_GRAPH_FULL_CHAR__ 'O'
// #define __INTERVAL_GRAPH_PART_CHAR__ 'o'
// #define __INTERVAL_GRAPH_NOT_CHAR__ '.'
#define __POINT_CONDITION_SEPARATOR__ ';'


// #define __ch2_condition_SEPARATOR__ ','


bool				ch_condition::init = false;
unsigned long		ch_condition::no_inputs;
double		        ch_condition::min_input;
double  	  		ch_condition::max_input;
double  			ch_condition::r0;
double  			ch_condition::m0;
ch_condition::t_crossover_type 	ch_condition::crossover_type;
bool				ch_condition::flag_use_hull = false;
bool				ch_condition::flag_margins = false;

// const std::vector<std::string> ch_condition::configuration_parameters; // = {"condition size", "dontcare probability", "mutate with dontcare","crossover","mutation"};

ch_condition::ch_condition()
{
	if (!ch_condition::init)
	{
		xcs_utility::error(class_name(),"ch_condition()", "not inited", 1);
	}
	// value.reserve(no_inputs);
	ch_values.reserve(no_inputs);
	hull.reserve(no_inputs);
}

ch_condition::ch_condition(const ch_condition& condition)
{
	//! check whether class has been initialized
	if (!ch_condition::init)
	{
		xcs_utility::error(class_name(),"ch_condition(const ch_condition&)", "not inited", 1);
	}
	ch_values = condition.ch_values;
	hull = condition.hull;
}

ch_condition::ch_condition(xcslib::configuration_manager& xcs_config)
{
	string	str_mutation_type;
	string	str_crossover_type;

	if (!ch_condition::init)
	{
		if (!xcs_config.exist(tag_name()))
		{
			xcs_utility::error(class_name(), "constructor", "section <" + tag_name() + "> not found", 1);	
		}

		xcs_config.check_parameters(tag_name(),supported_configuration_parameters);
		
		try {
			no_inputs = xcs_config.Value(tag_name(), "number of points");
			if (no_inputs<3)
			{
				string msg = "number of points must be >= 3";
				xcs_utility::error(class_name(), "constructor", msg, 1);
			}
		} catch (...) {
			string attribute = "number of points";
			xcs_utility::error(class_name(), "constructor", "attribute \'" + string(attribute) + "\' not found in <" + tag_name() + ">", 1);
		}


		try {
			r0 = xcs_config.Value(tag_name(), "r0");
		} catch (...) {
			string attribute = "r0";
			xcs_utility::error(class_name(), "constructor", "attribute \'" + string(attribute) + "\' not found in <" + tag_name() + ">", 1);
		}
		
		try {
			m0 = xcs_config.Value(tag_name(), "m0");
		} catch (...) {
			string attribute = "m0";
			xcs_utility::error(class_name(), "constructor", "attribute \'" + string(attribute) + "\' not found in <" + tag_name() + ">", 1);
		}

		//! parameters with default values
		str_crossover_type = (string) xcs_config.Value(tag_name(), "crossover", "one-pont");
		min_input = xcs_config.Value(tag_name(), "min input", 0.0);
		max_input = xcs_config.Value(tag_name(), "max input", 1.0);
		xcs_utility::set_flag(xcs_config.Value(tag_name(), "margins","off"),flag_margins);
		xcs_utility::set_flag(xcs_config.Value(tag_name(), "genetic operators use hull","off"),flag_use_hull);
	}
	init = true;
}

bool
ch_condition::operator<(const ch_condition& cond) const
{
	return (string_value()<cond.string_value());
};

string
ch_condition::string_value()
const
{
	vector<coordinate>	pts;

	if (!flag_use_hull)
	{
		pts = ch_values;
	} else { 
		pts = hull;
	}	
	sort(pts.begin(), pts.end());
	
	int sz = size();
	ostringstream out;
	for (int i=0; i<sz; i++)
	{
		if (i!=sz-1)
			out << pts[i].string_value() << __POINT_CONDITION_SEPARATOR__;
		else
			out << pts[i].string_value();
	}
	return out.str();
}

void
ch_condition::set_string_value(string str)
{
	ch_values.clear();
	istringstream	IN(str);
	double	x, y;
	char cm;
	char sm;

	int	old_pos = 0;
	int	pos;
	
	pos = str.find(';');	
	while (pos != string::npos)
	{
		string point(str,old_pos,(pos-old_pos));
		
		int posx = point.find(',');
		
		string xs = point.substr(0, posx);
		string ys = point.substr(posx+1);
		
		double x = atof(xs.c_str());
		double y = atof(ys.c_str());
		
		coordinate pt(x,y);
		ch_values.push_back(pt);	
		old_pos = pos+1;
		
		pos = str.find(';', old_pos);
	}
	
	if (pos==string::npos)
	{
		string point(str,old_pos,(pos-old_pos));
		int posx = point.find(',');
		
		string xs = point.substr(0, posx);
		string ys = point.substr(posx+1);
		
		//cout << "X=" << xs << " Y=" << ys << endl;
		
		double x = atof(xs.c_str());
		double y = atof(ys.c_str());
		
		//cout << "X=" << x << " Y=" << y << endl;
		
		coordinate pt(x,y);
		ch_values.push_back(pt);	
	}
	hull = compute_convex_hull(ch_values);
	// compute_generality(hull);
}

bool
ch_condition::operator==(const ch_condition& condition)
const
{
	for (int i=0; i <no_inputs; i++)
	{
		if (ch_values[i]!=condition.ch_values[i])
			return false;
	}
	return true;
};

bool
ch_condition::operator!=(const ch_condition& condition)
const
{
	return !((*this)==condition);
	// for (int i=0; i <no_inputs; i++)
	// {
	// 	if (value[i]!=cond.value[i])
	// 		return true;
	// }
	// return false;
}

ch_condition&
ch_condition::operator=(const ch_condition& condition)
{
	ch_values.clear();
	hull.clear();
	
	ch_values = condition.ch_values;
	hull = condition.hull;
	generality_ = condition.generality_;
	
	assert(ch_values.size()==condition.ch_values.size());
	return (*this);
}

void
ch_condition::normalize(vector<double>& input, vector<double>& norm_input)
{
	assert(false);
}

bool
ch_condition::match(const real_inputs& inputs) const
{
	assert(inputs.size()==2);

	xcslib::coordinate pti;
	pti.x = inputs.input(0);
	pti.y = inputs.input(1);

    bool fp = false;
    bool mp = false;

    // the point is one of the hull vertices
    if (std::find(hull.begin(),hull.end(),pti)!=hull.end())
    {
        return true;
    }

    return inside(hull,pti);

}

void
ch_condition::cover(const real_inputs& inputs)
{
	//! ch_condition applies only to 2d problems
	if (inputs.size()!=2)
	{
		xcs_utility::error(class_name(),"cover(...)", "can be applied only to two no_inputsensional problems", 1);
	}

	//! simple cover condition consisting of n points
	cover_with_points(inputs);	
	
	assert(ch_values.size()==no_inputs);
	
	if (flag_margins)
		set_bound(ch_values);
	
	hull = compute_convex_hull(ch_values);
}

void
ch_condition::set_bound(vector<coordinate> &v)
{	
	for(unsigned int p=0; p<v.size(); p++)
	{
		v[p].x = std::max(v[p].x,min_input);
		v[p].x = std::min(v[p].x,max_input);
		v[p].y = std::max(v[p].y,min_input);
		v[p].y = std::min(v[p].y,max_input);
	}
}

void	
ch_condition::cover_with_points(const real_inputs& s)
{
	//! ch_condition applies only to 2d problems
	if (s.size()!=2)
	{
		xcs_utility::error(class_name(),"cover(...)", "can be applied only to two no_inputsensional problems", 1);
	}

	//! 
	double xc = s.input(0);
	double yc = s.input(1);

	coordinate pti(xc,yc);
	
	vector<double>		angle;
	vector<double>		distance;
	vector<coordinate>	pts;
	vector<coordinate>	hull;

	angle.clear();
	distance.clear();
	pts.clear();

	for(unsigned long i=0; i<no_inputs; i++)
	{
		angle.push_back(xcs_random::random()*2*M_PI);
		distance.push_back(xcs_random::random()*ch_condition::r0);
		//cout << "ROT " << angle[i] << "\t" << distance[i] << endl;
	}
	
	for(unsigned long i=0; i<no_inputs; i++)
	{
		coordinate pt;
		pt.x = xc + distance[i]*cos(angle[i]);
		pt.y = yc + distance[i]*sin(angle[i]);
		//cout << "POINT " << pt.string_value() << endl;
		pts.push_back(pt);
	}
	
	hull = compute_convex_hull(pts);
	
	if (!inside(hull, pti))
	{
		int replace = xcs_random::dice(no_inputs);
		pts[replace] = pti;	
		hull = compute_convex_hull(pts);
	}
	
	assert(inside(hull, pti));	
	ch_values = pts;
}

void
ch_condition::mutate(double mutation_rate, const real_inputs &inputs)
{
	mutate(mutation_rate);
}

void
ch_condition::mutate(double mutation_rate)
{
	vector<coordinate> result;

	if (flag_use_hull)
	{
		result = mutate_vector(hull,mutation_rate);

		//! bound the points to the domain if required
		if (flag_margins)
			set_bound(result);
		
		hull = compute_convex_hull(result);
	}
	else 
	{
		ch_values = mutate_vector(ch_values,mutation_rate);

		//! bound the points to the domain if required
		if (flag_margins)
			set_bound(ch_values);
		
		hull = compute_convex_hull(ch_values);
	}
}

vector<coordinate> 
ch_condition::mutate_vector(const vector<coordinate> &points, double mutation_rate) const
{
	vector<coordinate> result = points;

	for (size_t points_i=0; points_i<result.size(); points_i++)
	{
		//! mutate the x coordinate
		if (xcs_random::random()<mutation_rate)
		{
			result[points_i].x += xcs_random::random()*m0*xcs_random::sign();
		}
		
		//! mutate the y coordinate
		if (xcs_random::random()<mutation_rate)
		{
			result[points_i].y += xcs_random::random()*m0*xcs_random::sign();
		}
	}
	return result;
}

void
ch_condition::recombine(ch_condition& offspring)
{
	recombine(offspring, crossover_type);
}

void
ch_condition::recombine(ch_condition& offspring, t_crossover_type crossover_type)
{	
	vector<coordinate> r1;
	vector<coordinate> r2;

	if (flag_use_hull)
	{
		r1 = hull;
		r2 = offspring.hull;

		one_point_crossover(r1,r2);

		if (flag_margins)	
		{
			set_bound(r1);
			set_bound(r2);			
		}

		hull = compute_convex_hull(r1);
		offspring.hull = compute_convex_hull(r1);

	} else {
		r1 = ch_values;
		r2 = offspring.ch_values;

		one_point_crossover(r1,r2);

		if (flag_margins)	
		{
			set_bound(r1);
			set_bound(r2);			
		}

		ch_values = r1;
		hull = compute_convex_hull(r1);

		offspring.ch_values = r2; 
		offspring.hull = compute_convex_hull(r2);
	}
}

// void 
// ch_condition::recombine_vectors(vector<coordinate> &r1, vector<coordinate> &r2)
// {
// 	switch (crossover_type)
// 	{
// 		case t_crossover_type::one_point:
// 			one_point_crossover(v1, v2, r1, r2);
// 			break;
// 		case t_crossover_type::uniform:
// 			uniform_crossover(v1, v2, r1, r2);
// 			break;

// 	}	
// }

//! uniform crossover
void
ch_condition::uniform_crossover(vector<coordinate> &r1, vector<coordinate> &r2)
{
 	assert(r1.size()==r2.size());
	
 	for(unsigned long it=0; it<r1.size(); it++)
 	{
 		if (xcs_random::random()<.5)
		{
			swap(r1[it].x,r2[it].x);
		}
		
		if (xcs_random::random()<.5)
		{
			//! swap y
			swap(r1[it].x,r2[it].x);
		}			
 	}
}

void
ch_condition::one_point_crossover(vector<coordinate> &v1, vector<coordinate> &v2)
{
	if ((v1.size()==1) || (v2.size()==1))
		return;

	unsigned long pos1;	//! first cut point	
	unsigned long pos2;	//! second cut point

	vector<coordinate> o1;
	vector<coordinate> o2;

	pos1 = xcs_random::dice(v1.size()-1)+1;
	pos2 = xcs_random::dice(v2.size()-1)+1;

	//! build first offspring
	for(unsigned long pt1=0; pt1<pos1; pt1++)
		o1.push_back(v1[pt1]);
	for(unsigned long pt1=pos2; pt1<v2.size(); pt1++)
		o1.push_back(v2[pt1]);

	//! build second offspring
	for(unsigned long pt2=0; pt2<pos2; pt2++)
		o2.push_back(v2[pt2]);
	for(unsigned long pt2=pos1; pt2<v1.size(); pt2++)
		o2.push_back(v1[pt2]);

	v1 = o1;
	v2 = o2;
}


bool
ch_condition::is_more_general_than(const ch_condition& cond)
const
{
	bool result = true;
	for(int i=0; result && (i<cond.ch_values.size()); i++)
	{
		result = inside(hull, cond.ch_values[i]);
	}
	return result;
}

void
ch_condition::random()
{
	ch_values.clear();
	
	double xc = min_input+xcs_random::random()*(max_input-min_input);
	double yc = min_input+xcs_random::random()*(max_input-min_input);
	
	ostringstream	in;
	
	in << xc << ' ' << yc;
	
	real_inputs s(in.str());
	
	assert(s.string_value()==in.str());
	
	cover(s);
}

// //! uniform crossover
// void
// ch_condition::uniform_crossover(ch_condition& offspring)
// {
//  	unsigned long sz = size();
	
//  	for(unsigned long it=0; it<sz; it++)
//  	{
//  		if (xcs_random::random()<.5)
// 		{
// 			//! swap x
// 			swap(ch_values[it].x, offspring.ch_values[it].x);
// 		}
		
// 		if (xcs_random::random()<.5)
// 		{
// 			//! swap y
// 			swap(ch_values[it].y, offspring.ch_values[it].y);
// 		}			
//  	}
// }


//! single point crossover
// void
// ch_condition::single_point_crossover(ch_condition& offspring)
// {
// 	assert( ch_values.size()==offspring.ch_values.size() );

// 	unsigned long sz = ch_values.size();		//! the allele are twice the number of points in the condition
// 	unsigned long pos;

// 	//! if size is one, just swap, no recombination
// 	if (sz!=1)
// 		pos = xcs_random::dice(sz-1)+1;
// 	else
// 		pos=0;

// 	//! selects which part of the point should be swapped, the whole (0), only the y (1), or nothing (2)
// 	int end_pos = xcs_random::dice(3);
// 	switch (end_pos) {
// 		case 0:
// 			swap(ch_values[pos],offspring.ch_values[pos]);
// 			break;
// 		case 1:
// 			swap(ch_values[pos].y, offspring.ch_values[pos].y);
// 			break;
// 	}

// 	for (int i=pos+1; i<no_inputs; i++)
// 	{
// 		swap(ch_values[i],offspring.ch_values[i]);
// 	}
	
// 	assert(ch_values.size()==no_inputs);
// 	assert(offspring.ch_values.size()==no_inputs);
// }


// void 
// ch_condition::set_crossover_type(string str)
// const
// {
// 	if (str.compare("one-point")==0)
// 		crossover_type = t_crossover_type::one_point;
// 	else if (str.compare("two-point")==0)
// 		crossover_type = t_crossover_type::two_point;
// 	else if (str.compare("uniform")==0)
// 		crossover_type = t_crossover_type::uniform;
// 	else
// 		xcs_utility::error("ch2_condition","set_crossover_type","unrecognized crossover type",1);
// };

double
ch_condition::compute_generality(const vector<coordinate> &pts) 
const
{
	double gen = 0;

	coordinate pt1 = pts[pts.size()-1];
	coordinate pt2 = pts[0];
	
	if (pts.size()==0) return 0;
	if (pts.size()==1) return 0;


	gen += euclidean_distance(pt1, pt2);

	for (unsigned i=1; i<pts.size(); i++)
	{	
		pt1 = pt2; 
		pt2 = pts[i];
		gen += euclidean_distance(pt1, pt2);
	}
	assert(gen<=4);
	return gen;
};

void ch_condition::print_parameters(ostream &OUTPUT) const
{
	OUTPUT << "<" << tag_name() << ">" << endl;
	OUTPUT << "\t" << "number of points = " << no_inputs << endl;
	OUTPUT << "\t" << "min input = " << min_input << endl;
	OUTPUT << "\t" << "max input = " << max_input << endl;
	OUTPUT << "\t" << "r0 = " << r0 << endl;
	OUTPUT << "\t" << "m0 = " << m0 << endl;

	map<t_crossover_type,string>::const_iterator pos = crossover_type_to_string.find(crossover_type);
	OUTPUT << "\t" << "crossover = " << pos->second << endl;
	
	OUTPUT << "\t" << "margins = " << (flag_margins?"on":"off") << endl;
	OUTPUT << "\t" << "genetic operators use hull points = " << (flag_use_hull?"on":"off") << endl;
	OUTPUT << "</" << tag_name() << ">" << endl;
};

vector<xcslib::coordinate> 
ch_condition::compute_convex_hull(const std::vector<xcslib::coordinate> &points) 
const
{

    // Store number of points points
    int n = points.size();

    // Convex hull is not possible if there are fewer than 3 points
    if (n < 3) return points;

    // Convert points 2D vector into vector of Point structures
    vector<xcslib::coordinate> a = points;

    // Find the point with the lowest y-coordinate (and leftmost in case of tie)
    xcslib::coordinate p0 = *min_element(a.begin(), a.end(), [](xcslib::coordinate a, xcslib::coordinate b) {
        return make_pair(a.y, a.x) < make_pair(b.y, b.x);
    });

    // Sort points based on polar angle with respect to the reference point p0
    sort(a.begin(), a.end(), [&p0](const xcslib::coordinate& a, const xcslib::coordinate& b) {
        int o = orientation(p0, a, b);

        // If points are collinear, keep the farthest one last
        if (o == 0) {
            return euclidean_distance(p0, a) < euclidean_distance(p0, b);
        }

        // Otherwise, sort by counter-clockwise order
        return o < 0;
    });

    // Vector to store the points on the convex hull
    vector<xcslib::coordinate> st;

    // Process each point to build the hull
    for (int i = 0; i < (int)a.size(); ++i) {

        // While last two points and current point make a non-left turn, remove the middle one
        while (st.size() > 1 && orientation(st[st.size() - 2], st.back(), a[i]) >= 0)
            st.pop_back();

        // Add the current point to the hull
        st.push_back(a[i]);
    }

    // If fewer than 3 points in the final hull, return {-1}
    if (st.size() < 3) return vector<xcslib::coordinate>();

    // we duplicate the start so that we can draw it and match it with no special case
    st.push_back(st[0]);
    return st;
}

bool
ch_condition::inside(const std::vector<xcslib::coordinate> &hull, const xcslib::coordinate &point)
const
{
    if (hull.size()==0) return false;
    if (hull.size()==1) return point==hull[0];

    bool inside = true;

    for (unsigned i = 1; inside && (i<hull.size()); i++)
    {
        inside = right_turn(hull[i-1], hull[i], point);
    }
    return inside;
}