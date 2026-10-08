#include "code_fragment.h"

using namespace xcslib;

unsigned long code_fragment::id_generator = 0;
unsigned long code_fragment::cf_max_length;
bool code_fragment::init = false;
unsigned long input_size;
vector<code_fragment::cf_operator> code_fragment::available_functions
{
                cf_operator::NOT, 
                cf_operator::NOR, 
                cf_operator::NAND, 
                cf_operator::OR, 
                cf_operator::AND};