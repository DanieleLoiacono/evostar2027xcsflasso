#ifndef __PREDICTION_FUNCTIONS__
#define __PREDICTION_FUNCTIONS__

#include "pf/base.h"
#include "pf/utility.h"

//! piecewise-linear prediction function
#include "pf/nlms.h"
#ifndef XCSF_NLMS_ONLY
#include "pf/value.h"
#include "pf/rls.h"
#include "pf/rlsk.h"
#include "pf/rls_delta.h"
#endif
#endif
