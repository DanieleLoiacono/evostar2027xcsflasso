// Predictor-level test driver for the xcslib prediction functions.
// Campaign tool (NOT part of xcslib): linked by scripts/02_build_cxx.sh against the same object
// files as xcsf-rf (every object except xcsf_main), and used by `xcsfcamp validate` to check the
// C++ update rules sample by sample against NumPy transcriptions and the Python mapping.
//
// Usage : pf_driver <confsys-suffix> [clone_at]     reads confsys.<suffix> in the current directory
// stdin : one sample per line "x1 ... xn y"
// stdout: one line per sample: prediction for the same input AFTER the update, then the weights
//         as printed by the prediction function ("[w0;w1;...]"), 17 significant digits.
// clone_at = k > 0: after the k-th update the function is replaced by its clone()
//         (offspring semantics: weights inherited, internal state re-initialised).
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "configuration_manager.h"
#include "prediction_functions.h"

int main(int argc, char **argv)
{
	if (argc < 2) { std::cerr << "usage: pf_driver <suffix> [clone_at]" << std::endl; return 2; }
	const long clone_at = argc > 2 ? std::atol(argv[2]) : 0;
	xcslib::configuration_manager cfg(argv[1]);
	xcsflib::init_prediction_functions(cfg);
	xcsflib::base_pf *f = xcsflib::get_prediction_function(NULL);
	const unsigned long d = f->dim();
	std::cout << std::setprecision(17);
	std::string line;
	long t = 0;
	while (std::getline(std::cin, line))
	{
		std::istringstream is(line);
		std::vector<double> x(d);
		double y;
		for (unsigned long i = 0; i < d; i++) is >> x[i];
		if (!(is >> y)) continue;
		f->update(x, y);
		t++;
		std::cout << f->output(x) << " ";
		f->print(std::cout);
		std::cout << "\n";
		if (clone_at > 0 && t == clone_at)
		{
			xcsflib::base_pf *g = f->clone(NULL);
			delete f;
			f = g;
		}
	}
	delete f;
	return 0;
}
