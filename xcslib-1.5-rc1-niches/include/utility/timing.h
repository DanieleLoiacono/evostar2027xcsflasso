#include <iostream>
#include <ostream>
#include <vector>
#include <numeric>
#include <iterator>
#include <algorithm>

#include "high_resolution_timer.h"

using namespace std;

#ifndef __XCSLIB_TIMING__
#define __XCSLIB_TIMING__

namespace xcslib
{
    class timing
    {
        public:
            timing() {}

            void clear()
            {
                learning_problems_time.clear();
                learning_problems_steps.clear();
                testing_problems_time.clear();	
                testing_problems_steps.clear();	
                condensation_problems_time.clear();
                condensation_problems_steps.clear();
                average_learning_problems_step_time.clear();
                average_testing_problems_step_time.clear();
                average_condensation_problems_step_time.clear();
            }

            void update(double elapsed_time, 
                bool is_learning_problem,
                bool is_condensation_problem,
                unsigned long no_steps)
                {
                    if (is_learning_problem)
                    {
                        if (is_condensation_problem)
                        {
                            condensation_problems_time.push_back(elapsed_time);
                            condensation_problems_steps.push_back(no_steps);
                        } else {
                            learning_problems_time.push_back(elapsed_time);
                            learning_problems_steps.push_back(no_steps);
                        }                    
                    } else {
                            testing_problems_time.push_back(elapsed_time);
                            testing_problems_steps.push_back(no_steps);
                    }
                }
            
            void save_report(ostream &output, string label="") const
            {
                output << label << "\t# Learning Problems = " << learning_problems_time.size() << endl;
                output << label << "\t# Condensation Problems = " << condensation_problems_time.size() << endl;
                output << label << "\t# Testing Problems = " << testing_problems_time.size() << endl;
                output << endl;

                output << label << "\tTotal Elapsed Time = " << total_learning_problem_time + total_condensation_problem_time + total_testing_problem_time << endl;
                output << label << "\tTotal Number of Steps = " << total_learning_problem_steps + total_condensation_problem_steps + total_testing_problem_steps << endl;
                output << endl;

                output << label << "\tTotal Elapsed Time for Learning = " << total_learning_problem_time << endl;
                output << label << "\tTotal Steps for Learning = " << total_learning_problem_steps << endl;
                output << label << "\tOverall Average Time for a Learning Problem = " << overall_average_learning_problems_step_time << endl;
                output << label << "\tMean Time for a Learning Problem = " << learning_problem_time_mean << endl;
                output << label << "\tStdev Time for a Learning Problem = " << learning_problem_time_std << endl;
                output << label << "\tMean Time for a Learning Step = " << average_learning_problems_step_time_mean << endl;
                output << label << "\tStdev Time for a Learning Step = " << average_learning_problems_step_time_std << endl;
                output << endl;                

                output << label << "\tTotal Elapsed Time for Testing = " << total_testing_problem_time << endl;
                output << label << "\tTotal Steps for Testing = " << total_testing_problem_steps << endl;
                output << label << "\tOverall Average Time for a Testing Problem = " << overall_average_testing_problems_step_time << endl;
                output << label << "\tMean Time for a Testing Problem = " << testing_problem_time_mean << endl;
                output << label << "\tStdev Time for a Testing Problem = " << testing_problem_time_std << endl;
                output << label << "\tMean Time for a Testing Step = " << average_testing_problems_step_time_mean << endl;
                output << label << "\tStdev Time for a Testing Step = " << average_testing_problems_step_time_std << endl;
                output << endl;   

                output << label << "\tTotal Elapsed Time for Condensation = " << total_condensation_problem_time << endl;
                output << label << "\tTotal Steps for Condensation = " << total_condensation_problem_steps << endl;
                output << label << "\tOverall Average Time for a Condensation Problem = " << overall_average_condensation_problems_step_time << endl;
                output << label << "\tMean Time for a Condensation Problem = " << condensation_problem_time_mean << endl;
                output << label << "\tStdev Time for a Condensation Problem = " << condensation_problem_time_std << endl;
                output << label << "\tMean Time for a Condensation Step = " << average_condensation_problems_step_time_mean << endl;
                output << label << "\tStdev Time for a Condensation Step = " << average_condensation_problems_step_time_std << endl;
                output << endl;                
            }

            void save_data(ostream &output) const
            {

            }

            void compute_statistics()
            {
                init_statistics();

                if (learning_problems_time.size()>0)
                {
                    total_learning_problem_time = std::accumulate(learning_problems_time.begin(), learning_problems_time.end(), 0);
                    total_learning_problem_steps = std::accumulate(learning_problems_steps.begin(), learning_problems_steps.end(), 0);

                    //! average time for one learning step
                    overall_average_learning_problems_step_time = total_learning_problem_time/total_learning_problem_steps;

                    //! average time for one problem
                    learning_problem_time_mean = mean(learning_problems_time);
                    learning_problem_time_std = variance(learning_problems_time);

                    for (size_t i=0; i<learning_problems_time.size(); i++)
                    {
                        average_learning_problems_step_time.push_back(learning_problems_time[i]/learning_problems_steps[i]);
                    }

                    average_learning_problems_step_time_mean = mean(average_learning_problems_step_time);
                    average_learning_problems_step_time_std = variance(average_learning_problems_step_time);
                }

                if (testing_problems_time.size())
                {
                    total_testing_problem_time = std::accumulate(testing_problems_time.begin(), testing_problems_time.end(), 0);
                    total_testing_problem_steps = std::accumulate(testing_problems_steps.begin(), testing_problems_steps.end(), 0);                

                    //! average time for one testing step
                    overall_average_testing_problems_step_time = total_testing_problem_time/total_testing_problem_steps;

                    testing_problem_time_mean = mean(testing_problems_time);
                    testing_problem_time_std = variance(testing_problems_time);

                    for (size_t i=0; i<testing_problems_time.size(); i++)
                        average_testing_problems_step_time.push_back(testing_problems_time[i]/testing_problems_steps[i]);

                    average_testing_problems_step_time_mean = mean(average_testing_problems_step_time);
                    average_testing_problems_step_time_std = variance(average_testing_problems_step_time);
                }

                if (condensation_problems_time.size()>0)
                {
                    total_condensation_problem_time = std::accumulate(condensation_problems_time.begin(), condensation_problems_time.end(), 0);
                    total_condensation_problem_steps = std::accumulate(condensation_problems_steps.begin(), condensation_problems_steps.end(), 0);

                    //! average time for one condensation step
                    overall_average_condensation_problems_step_time = total_condensation_problem_time/total_condensation_problem_steps;

                    condensation_problem_time_mean = mean(condensation_problems_time);        
                    condensation_problem_time_std = variance(condensation_problems_time);

                    for (size_t i=0; i<condensation_problems_time.size(); i++)
                    {
                        average_condensation_problems_step_time.push_back(condensation_problems_time[i]/condensation_problems_steps[i]);
                    }
                    average_condensation_problems_step_time_mean = mean(average_condensation_problems_step_time);
                    average_condensation_problems_step_time_std = variance(average_condensation_problems_step_time);
                }
            }

        private:
            // xcslib::high_resolution_timer timer_problem(xcslib::high_resolution_timer::resolution::microseconds);
            vector<double>	learning_problems_time;
            vector<double>	learning_problems_steps;
            vector<double>	testing_problems_time;	
            vector<double>	testing_problems_steps;	
            vector<double>	condensation_problems_time;
            vector<double>	condensation_problems_steps;

            double total_learning_problem_time;
            double total_testing_problem_time;
            double total_condensation_problem_time;
            double total_learning_problem_steps;
            double total_testing_problem_steps;
            double total_condensation_problem_steps;

            double learning_problem_time_mean;
            double learning_problem_time_std;

            double testing_problem_time_mean;
            double testing_problem_time_std;

            double condensation_problem_time_mean;
            double condensation_problem_time_std;

            //! overall average computed as the total_time/total_no_steps
            double overall_average_learning_problems_step_time;
            double overall_average_testing_problems_step_time;
            double overall_average_condensation_problems_step_time;

            //! time spent to solve a problem divided by the number of steps to solve the problem
            vector<double>	average_learning_problems_step_time;
            vector<double>	average_testing_problems_step_time;
            vector<double>	average_condensation_problems_step_time;

            //! mean and standard deviation of the single averages (provides a confidence interval)
            double average_learning_problems_step_time_mean;
            double average_testing_problems_step_time_mean;
            double average_condensation_problems_step_time_mean;

            double average_learning_problems_step_time_std;
            double average_testing_problems_step_time_std;
            double average_condensation_problems_step_time_std;

            void init_statistics() 
            {
                total_learning_problem_time = 0;
                total_testing_problem_time = 0;
                total_condensation_problem_time = 0;
                total_learning_problem_steps = 0;
                total_testing_problem_steps = 0;
                total_condensation_problem_steps = 0;

                learning_problem_time_mean = 0;
                learning_problem_time_std = 0;

                testing_problem_time_mean = 0;
                testing_problem_time_std = 0;

                condensation_problem_time_mean = 0;
                condensation_problem_time_std = 0;

                //! overall average computed as the total_time/total_no_steps
                overall_average_learning_problems_step_time = 0;
                overall_average_testing_problems_step_time = 0;
                overall_average_condensation_problems_step_time = 0;

                //! mean and standard deviation of the single averages (provides a confidence interval)
                average_learning_problems_step_time_mean = 0;
                average_learning_problems_step_time_std = 0;

                average_testing_problems_step_time_mean = 0;
                average_testing_problems_step_time_std = 0;

                average_condensation_problems_step_time_mean = 0;
                average_condensation_problems_step_time_std = 0;                
            }            

        template <typename Container, typename T = typename std::decay<decltype(*std::begin(std::declval<Container>()))>::type>
        T variance(Container && c)
        {
            auto b = std::begin(c), e = std::end(c);
            auto size = std::distance(b, e);
            auto sum = std::accumulate(b, e, T());
            auto mean = sum / size;
            T accum = T();
            for (const auto d : c)
                accum += (d - mean) * (d - mean);
            return std::sqrt(accum / (size - 1));
        }            

        template <typename Container, typename T = typename std::decay<decltype(*std::begin(std::declval<Container>()))>::type>
        T mean(Container && c)
        {
            auto b = std::begin(c), e = std::end(c);
            auto size = std::distance(b, e);
            auto sum = std::accumulate(b, e, T());
            return sum / size;
        }            

        template <typename Container, typename T = typename std::decay<decltype(*std::begin(std::declval<Container>()))>::type>
        T sum(Container && c)
        {
            auto b = std::begin(c), e = std::end(c);
            auto size = std::distance(b, e);
            return std::accumulate(b, e, T());
        }
    };
}

#endif