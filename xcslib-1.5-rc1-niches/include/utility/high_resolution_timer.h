#include <chrono>

#ifndef __XCSLIB_HR_TIMER__
#define __XCSLIB_HR_TIMER__

namespace xcslib
{
   //! timer class to measure CPU execution time 
   class high_resolution_timer 
   {
      public:
         enum class resolution 
         {
            milliseconds,
            microseconds,
            nanoseconds
         };

      private:
         std::chrono::time_point<std::chrono::high_resolution_clock> start_time;	//! init time
         std::chrono::time_point<std::chrono::high_resolution_clock> final_time;	//! stop time
         resolution default_resolution = resolution::milliseconds;

      public:
         high_resolution_timer() { };
         high_resolution_timer(resolution default_resolution_): default_resolution(default_resolution_) { };
      
         ~high_resolution_timer() {};

         void start_timer()
         {	
            start_time = std::chrono::high_resolution_clock::now();
         }

         void stop_timer()
         {	
            final_time = std::chrono::high_resolution_clock::now();
         }

         double elapsed(high_resolution_timer::resolution time_resolution) const
         {
            switch (time_resolution)
            {
                case resolution::milliseconds:
                    return std::chrono::duration<double, std::milli>(final_time - start_time).count();

                case resolution::microseconds:
                    return std::chrono::duration<double, std::micro>(final_time - start_time).count();

                case resolution::nanoseconds:
                    return std::chrono::duration<double, std::nano>(final_time - start_time).count();

            }            
         }

         unsigned long elapsedl(high_resolution_timer::resolution time_resolution) const
         {
            switch (time_resolution)
            {
                case resolution::milliseconds:
                    return std::chrono::duration_cast<std::chrono::milliseconds>(final_time - start_time).count();

                case resolution::microseconds:
                    return std::chrono::duration_cast<std::chrono::microseconds>(final_time - start_time).count();

                case resolution::nanoseconds:
                    return std::chrono::duration_cast<std::chrono::nanoseconds>(final_time - start_time).count();

            }            
         }

         double elapsed() const 
         {
            return elapsed(default_resolution);
         }

         unsigned long elapsedl() const 
         {
            return elapsedl(default_resolution);
         }
   };
}
#endif