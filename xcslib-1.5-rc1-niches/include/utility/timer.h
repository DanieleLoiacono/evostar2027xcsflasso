#include <ctime>
#include <unistd.h>
#include <sys/times.h>

#ifndef __XCSLIB_TIMER__
#define __XCSLIB_TIMER__

namespace xcslib
{
   //! timer class to measure CPU execution time 
   class timer {
      private:
         unsigned long ti;	//! init time
         unsigned long tf;	//! stop time
      
      public:
         timer() {
            struct tms reading;
            times(&reading);
            ti = reading.tms_utime;
         };
      
         ~timer() {};
         void start()
         {	 
            struct tms reading;
            times(&reading);
            ti = reading.tms_utime;
         }
         
         //! returns the time passed (seconds)
         double time() const
         {
            struct tms reading;
            times(&reading);
            return double(reading.tms_utime - ti)/sysconf(_SC_CLK_TCK);
         }
      
         void stop()
         {	 
            struct tms reading;
            times(&reading);
            tf = reading.tms_utime;
         }

         //! returns the elapsed time (seconds)
         double elapsed() const
         {
            return double(tf - ti)/sysconf(_SC_CLK_TCK);
         }

         unsigned long elapsed_ticks() const
         {
            return tf-ti;
         }
      
         unsigned long initial() const { return ti; };
         unsigned long final() const { return tf; };
   };
}

#endif