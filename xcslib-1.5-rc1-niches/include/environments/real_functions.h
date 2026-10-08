#ifndef __REAL_FUNCTIONS__
#define __REAL_FUNCTIONS__

#include <cassert>
#include <vector>

namespace xcslib 
{
    class real_functions
    {
        public:

        double sine(const std::vector <double>& current_inputs)
        {
            assert(current_inputs.size()==1); 
            return double(scale_factor)*sin((2*M_PI*current_inputs[0])/double(scale_factor));
        }

        // double sine3(const std::vector <double>& current_inputs, double scale_factor=1.0)
        // {
        //     assert(current_inputs.size()==1); 

        // }

        // double sine4(const std::vector <double>& current_inputs, double scale_factor=1.0)
        // {
        //     assert(current_inputs.size()==1); 
        // }

        // double abs(const std::vector <double>& current_inputs, double scale_factor=1.0)
        // {
        //     assert(current_inputs.size()==1); 
        // }
        
        // double abs2(const std::vector <double>& current_inputs, double scale_factor=1.0)
        // {
        //     assert(current_inputs.size()==1); 
        // }

        // double polynomial(const std::vector <double>& current_inputs, double scale_factor=1.0)
        // {
        //     assert(current_inputs.size()==1); 
        // }

        // double simfun(const std::vector <double>& current_inputs, double scale_factor=1.0)
        // {
        //     assert(current_inputs.size()==1); 
        // }

        // double simfun2(const std::vector <double>& current_inputs, double scale_factor=1.0)
        // {
        //     assert(current_inputs.size()==1); 

        // }

        // double simfun3(const std::vector <double>& current_inputs, double scale_factor=1.0)
        // {
        //     assert(current_inputs.size()==1); 

        // }

        double f1(const std::vector <double>& current_inputs, double scale_factor=1.0)
        {
            assert(current_inputs.size()==2); 
            return double(long(current_inputs[0]*3)%3)/3. + double(long(current_inputs[1]*3)%3)/3.;
        }

        double f2(const std::vector <double>& current_inputs, double scale_factor=1.0)
        {
            assert(current_inputs.size()==2); 
            return double(long((current_inputs[0]+current_inputs[1])*2)%4)/6.;
        }

        double f3(const std::vector <double>& current_inputs, double scale_factor=1.0)
        {
            assert(current_inputs.size()==2); 
            return sin(2*M_PI*(current_inputs[0]+current_inputs[1]));
        }

        double frog(const std::vector <double>& current_inputs, double scale_factor=1.0)
        {
            assert(current_inputs.size()==2); 

            double result;
            if (current_inputs[0]+current_inputs[1]<=1)
                result = current_inputs[0]+current_inputs[1];
            if (current_inputs[0]+current_inputs[1]>1)
                result = 2-(current_inputs[0]+current_inputs[1]);
            return result;
        }

        double hill(const std::vector <double>& current_inputs, double scale_factor=1.0)
        {
            assert(current_inputs.size()==2);
            double result;
            
            // f(x,y) = ((x-0.5)**2+(y-0.5)**2>=0.09) ? 0.28 : ((x-0.5)**2+(y-0.5)**2<0.09) ? (-8*((x-0.5)**2 + (y-0.5)**2))+1 : 1/0
            if (pow((current_inputs[0]-0.5),2)+pow((current_inputs[1]-0.5),2)>=0.09)
                result = 0.28;
            if (pow((current_inputs[0]-0.5),2)+pow((current_inputs[1]-0.5),2)<0.09)
                result = (-8*(pow((current_inputs[0]-0.5),2)+pow((current_inputs[1]-0.5),2)))+1;
            return result;

        }

        double step(const std::vector <double>& current_inputs, double scale_factor=1.0)
        {
            assert(current_inputs.size()==2); 
            double result;

            if (current_inputs[0]+current_inputs[1]>=1)
                result = 1;
            if (current_inputs[0]+current_inputs[1]<1)
                result = 0;
            return result;
        }
    }
}
#endif

// 	double compute_function(t_function selected_function, vector<double> current_inputs, double scale_factor)
// 	{
// 		double ris;
// 		switch (selected_function)
// 		{
// //////////////////////////////////////////////////////////////////////////////////////
// // 1 input functions /////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////////////////////////

// 			case FUNCTION_LINEAR:
// 				ris = double(scale_factor)*(3*(current_inputs[0]/double(scale_factor))+2);
// 				return ris;
// 				break;

// 			case FUNCTION_SINE:
// 				ris = double(scale_factor)*sin((2*M_PI*current_inputs[0])/double(scale_factor));
// 				return ris;
// 				break;
				
// 			case FUNCTION_SINE4:
// 				ris = double(scale_factor)* ( sin((2*M_PI*current_inputs[0])/double(scale_factor)) + \
// 					sin((4*M_PI*current_inputs[0])/double(scale_factor)) + \
// 					sin((6*M_PI*current_inputs[0])/double(scale_factor)) + \
// 					sin((8*M_PI*current_inputs[0])/double(scale_factor)) );
// 				return ris;
// 				break;
				
// 			case FUNCTION_SINE3:
// 				ris = double(scale_factor)* ( sin((2*M_PI*current_inputs[0])/double(scale_factor)) + \
// 					sin((4*M_PI*current_inputs[0])/double(scale_factor)) + \
// 					sin((6*M_PI*current_inputs[0])/double(scale_factor)) );
// 				return ris;
// 				break;

// 			case FUNCTION_ABS:
// 				ris = double(scale_factor)* fabs(sin((2*M_PI*current_inputs[0])/double(scale_factor)) + fabs(cos((2*M_PI*current_inputs[0])/double(scale_factor))));
// 				return ris;
// 				break;

// 			case FUNCTION_ABS2:
// // vecchia				current_reward = double(scale_factor)* fabs(sin((2*M_PI*current_input)/double(scale_factor)) + fabs(cos((2*M_PI*current_input)/double(scale_factor))));
// 				ris = double(scale_factor)* fabs(sin((2*M_PI*current_inputs[0])/double(scale_factor)) + fabs(cos((2*2*M_PI*current_inputs[0])/double(scale_factor))));
// 				return ris;
// 				break;

// //			case FUNCTION_SIXTIC:
// //				//current_reward = double(scale_factor)*(pow((current_input/double(scale_factor).),6)+ -2*pow((current_input/double(scale_factor).),4)+ pow((current_input/double(scale_factor).),2)); 
// //				break;

// 			case FUNCTION_POLYNOMIAL:	// 1+x+x^2+x^3  / SBAGLIATA!!! non sta nel range
// 				ris = double(scale_factor)*(1+(current_inputs[0]/double(scale_factor))+pow((current_inputs[0]/double(scale_factor)),2)+pow((current_inputs[0]/double(scale_factor)),3));	
// 				return ris;
// 				break;

// // AGGIUNTA ////////////////////////////////////////////////////////
// 			case FUNCTION_SIMFUN:	// 4x^2  0<=x<=0.5 :  -2x+2  0.5<=x<=1
// 				if ((current_inputs[0]/double(scale_factor))<=0.5)	
// 					ris = double(scale_factor)*(4*pow((current_inputs[0]/double(scale_factor)),2));
// 				if ((current_inputs[0]/double(scale_factor))>0.5)
// 					ris = double(scale_factor)*(((-2)*(current_inputs[0]/double(scale_factor)))+2);
// 				return ris;
// 				break;

// 			case FUNCTION_SIMFUN2:	// 8x^3  0<=x<=0.5 : -2x+2  0.5<=x<=1
// 				if ((current_inputs[0]/double(scale_factor))<=0.5)
// 					ris = double(scale_factor)*(8*pow((current_inputs[0]/double(scale_factor)),3));
// 				if ((current_inputs[0]/double(scale_factor))>0.5)
// 					ris = double(scale_factor)*(((-2)*(current_inputs[0]/double(scale_factor)))+2);
// 				return ris;
// 				break;

// 			case FUNCTION_SIMFUN3:	// 0.2  0<=x<=0.2 : 0.4  0.2<=x<=0.4 : 0.6  0.4<=x<=0.6 : 0.8  0.6<=x<=0.8 : 1  0.8<=x<=1
// 				if ((current_inputs[0]/double(scale_factor))<=0.2)				
// 					ris = double(scale_factor)*(0.2);
// 				if (((current_inputs[0]/double(scale_factor))>0.2) && ((current_inputs[0]/double(scale_factor))<=0.4))
// 					ris = double(scale_factor)*(0.4);
// 				if (((current_inputs[0]/double(scale_factor))>0.4) && ((current_inputs[0]/double(scale_factor))<=0.6))	
// 					ris = double(scale_factor)*(0.6);
// 				if (((current_inputs[0]/double(scale_factor))>0.6) && ((current_inputs[0]/double(scale_factor))<=0.8))
// 					ris = (0.8);
// 				if (((current_inputs[0]/double(scale_factor))>0.8) && ((current_inputs[0]/double(scale_factor))<=0.8))				
// 					ris = double(scale_factor)*(1);
// 				return ris;
// 				break;

// 			case FUNCTION_POLGRA1:	// x
				
// 				ris = double(scale_factor)*(current_inputs[0]/double(scale_factor));
// 				return ris;
// 				break;

// 			case FUNCTION_POLGRA2:	// x^2
// 				ris = double(scale_factor)*pow((current_inputs[0]/double(scale_factor)),2);
// 				return ris;
// 				break;

// 			case FUNCTION_POLGRA3:	// x^3
// 				ris = double(scale_factor)*pow((current_inputs[0]/double(scale_factor)),3);
// 				return ris;
// 				break;

// 			case FUNCTION_DOPPLER:	// vedi libro...
// 				ris = 24.2158 * sin((2*M_PI*(1+0.05))/(current_inputs[0]+0.05)) * sqrt(current_inputs[0]*(1-current_inputs[0]));
// 				return ris;
// 				break;

// 			case FUNCTION_HEAVISINE:	// vedi libro...
// 				double add1, add2;
// 				if ((current_inputs[0]-0.3)>=0)
// 					add1 = 1;
// 				else if ((current_inputs[0]-0.3)<0)
// 					add1 = -1;
// 				if ((0.72-current_inputs[0])>=0)
// 					add2 = 1;
// 				else if ((0.72-current_inputs[0])<0)
// 					add2 = -1;
// 				ris = 2.3564*((4*sin(4*M_PI*current_inputs[0])) - add1 - add2);
// 				return ris;
// 				break;


// //////////////////////////////////////////////////////////////////////////////////////
// // 2 inputs functions ////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////////////////////////

// 			case FUNCTION_F1:
// 				//f1 = mod(x*3,3)/3. + mod(y*3,3)/3.;
// 				ris = double(long(current_inputs[0]*3)%3)/3. + double(long(current_inputs[1]*3)%3)/3.;
// 				//current_reward = double(long(x*2)%2)/2. + double(long(y*2)%2)/2.;
// 				return ris;
// 				break;

// 			case FUNCTION_F2:
// 				//f1 = mod(x*3,3)/3. + mod(y*3,3)/3.;
// 				//f2 = mod((x+y)*2,4)/6.;
// 				ris = double(long((current_inputs[0]+current_inputs[1])*2)%4)/6.;
// 				return ris;
// 				break;

// 			case FUNCTION_F3:
// 				ris = sin(2*M_PI*(current_inputs[0]+current_inputs[1]));
// 				return ris;
// 				break;

// 			case FUNCTION_FROG:
// 				//cerr << "FROG " << current_inputs[0] << "," << current_inputs[1] << endl;
// 				if (current_inputs[0]+current_inputs[1]<=1)
// 					ris = current_inputs[0]+current_inputs[1];
// 				if (current_inputs[0]+current_inputs[1]>1)
// 					ris = 2-(current_inputs[0]+current_inputs[1]);
// 				return ris;
// 				break;

// // AGGIUNTA //////////////////////////////////////////////////////////////////
// 			case FUNCTION_STEP:
// 				// 
// 				if (current_inputs[0]+current_inputs[1]>=1)
// 					ris = 1;
// 				if (current_inputs[0]+current_inputs[1]<1)
// 					ris = 0;
// 				return ris;
// 				break;

// 			case FUNCTION_HILL:
// // f(x,y) = ((x-0.5)**2+(y-0.5)**2>=0.09) ? 0.28 : ((x-0.5)**2+(y-0.5)**2<0.09) ? (-8*((x-0.5)**2 + (y-0.5)**2))+1 : 1/0
// 				if (pow((current_inputs[0]-0.5),2)+pow((current_inputs[1]-0.5),2)>=0.09)
// 					ris = 0.28;
// 				if (pow((current_inputs[0]-0.5),2)+pow((current_inputs[1]-0.5),2)<0.09)
// 					ris = (-8*(pow((current_inputs[0]-0.5),2)+pow((current_inputs[1]-0.5),2)))+1;
// 				return ris;
// 				break;

// 			case FUNCTION_PLANE:
// 					ris = 0.5;
// 				return ris;
// 				break;



// //////////////////////////////////////////////////////////////////////////////////////
// // 3 inputs functions ////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////////////////////////

// 			case FUNCTION_F5:
// 				ris = sin(2*M_PI*(current_inputs[0]+current_inputs[1]+current_inputs[2]));
// 				return ris;
// 				break;



// //////////////////////////////////////////////////////////////////////////////////////
// // 4 inputs functions ////////////////////////////////////////////////////////////////
// //////////////////////////////////////////////////////////////////////////////////////

// 			case FUNCTION_F4:
// /*				cout << "COMPUTE f4(";
// 				cout << current_inputs[0];
// 				cout << ", " << current_inputs[1];
// 				cout << ", " << current_inputs[2];
// 				cout << ", " << current_inputs[3];
// 				cout << ") = " ;*/
// 				ris = current_inputs[1] + current_inputs[3] + (2*M_PI*sin(current_inputs[0])) + (2*M_PI*sin(current_inputs[2]));
// // 				cout << ris << endl;
// 				return ris;
// 				break;


// 			default:
// 				assert(false);
// 	    	}

// 	}
