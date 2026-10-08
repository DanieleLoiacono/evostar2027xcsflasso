/*!
 * \file rl_definitions.h
 *
 * \brief maps high level class names (e.g., t_condition) to the actual class name (e.g., xcs_bitstring_condition)
 *
 */

#ifndef __RL_DEFINITIONS__
#define __RL_DEFINITIONS__

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string.h>
#include <iomanip>

//! STL libraries
#include <algorithm>
#include <vector>

//! maximum number of char available for error messages
// const unsigned long MSGSTR = 2048;

//! maps the actual class used for detectors, specified with the __DETECTORS__ variable in 
//! the make file to the high level name t_sensors
class   __INPUTS__;
typedef __INPUTS__ t_state;

//! maps the actual class used for classifier actions, specified with the __ACTION__ variable in 
//! the make file to the high level name t_action
class   __ACTION__;
typedef __ACTION__ t_action;

//! maps the actual class used for the environment, specified with the __ENVIRONMENT__ variable in 
//! the make file to the high level name t_environment
class   __ENVIRONMENT__;
typedef __ENVIRONMENT__ t_environment;

#include __DET_INCLUDE__
#include __ENV_INCLUDE__
#include __ACT_INCLUDE__
	
extern t_environment*	Environment;
#endif
