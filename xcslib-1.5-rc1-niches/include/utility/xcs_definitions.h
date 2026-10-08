#define __XCSLIB_VERSION__ "1.3"

#ifndef __XCS_DEFINITIONS__
#define __XCS_DEFINITIONS__
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string.h>
#include <iomanip>

// STL libraries
#include <algorithm>
#include <vector>

#include "rl_definitions.h"

using namespace xcslib;

// maps the actual class used for classifier conditions, specified with the __CONDITION__ variable in 
// the make file to the high level name t_condition
class   __CONDITION__;
typedef __CONDITION__ t_condition;

#include __COND_INCLUDE__



//! maps the actual class used for the environment, specified with the __ENVIRONMENT__ variable in 
//! the make file to the high level name t_environment
#include __CLS_INCLUDE__

class   __CLASSIFIER__;
typedef __CLASSIFIER__ t_classifier;



//! maps the actual class used for the the classifier system, specified with the __MODEL__ variable in 
//! the make file to the high level name t_environment
#include __MOD_INCLUDE__
class   __MODEL__;
typedef __MODEL__ t_classifier_system;

#endif
