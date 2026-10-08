#include <string>
#include <fstream>
#include <algorithm>
#include "xcs_definitions.h"
#include "xcs_random.h"
#include "condition_base.h"
#include "cf_condition.h"

//! static variable declaration
bool cf_condition::init; //! true -> the class has been initialized
unsigned long cf_condition::no_code_fragments;
unsigned long cf_condition::code_fragment_size;
vector<code_fragment> cf_condition::cf_library;