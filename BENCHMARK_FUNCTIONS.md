# Continuous Benchmark Functions

## Key requirement

All benchmark domains are REAL and CONTINUOUS. Sample `x` using floating-point values from the stated distributions. Never enumerate only integer points.

The first functions are derived from the attached XCSF generalization paper, but here they are explicitly used as continuous functions.

## Core benchmark F1 - Sine Low

Name: `sine_low_1d`

Domain:

`x in [0, 100]`, sampled uniformly as a real number.

Target:

`f(x) = 100 * sin(2*pi*x/100)`

Purpose:

- simple smooth baseline;
- easy visual inspection;
- direct continuous analogue of the paper's discrete sine test.

## Core benchmark F2 - Sine Shifted

Name: `sine_shifted_1d`

Domain:

`x in [1000, 1100]`, sampled uniformly as a real number.

Target:

`f(x) = 100 * sin(2*pi*x/100)`

Purpose:

- same periodic shape and output range as F1;
- tests sensitivity to large input offsets/scales;
- useful for distinguishing predictor-update behavior.

This is a diagnostic benchmark, not an attempt to reproduce the paper's full input-range study.

## Core benchmark F3 - Sinus Three

Name: `sinus3_1d`

Domain:

`x in [950, 1050]`, sampled uniformly as a real number.

Target:

`f(x) = 100 * (sin(2*pi*x/100) + sin(4*pi*x/100) + sin(6*pi*x/100))`

Purpose:

- more complex smooth target;
- paper-derived;
- stresses the ability to form compact piecewise local approximations.

## Core benchmark F4 - Sinus Four

Name: `sinus4_1d`

Domain:

`x in [950, 1050]`, sampled uniformly as a real number.

Target:

`f(x) = 100 * (sin(2*pi*x/100) + sin(4*pi*x/100) + sin(6*pi*x/100) + sin(8*pi*x/100))`

Purpose:

- harder smooth target than F3;
- paper-derived;
- expected to make generalization differences more visible.

## Core benchmark F5 - Absolute Mixed Trigonometric Function

Name: `abs_mix_1d`

Domain:

`x in [950, 1050]`, sampled uniformly as a real number.

Target:

`f(x) = 100 * abs(sin(2*pi*x/100) + abs(cos(2*pi*x/100)))`

Purpose:

- paper-derived;
- introduces non-smooth structure through absolute values;
- complements the smooth sinusoidal targets.

## Optional extension F6 - Two-dimensional sinusoidal surface

Name: `sin_cos_surface_2d`

Domain:

`x1, x2 in [0,1]`, independently uniform real values.

Target:

`f(x1,x2) = 100 * sin(2*pi*x1) * cos(2*pi*x2)`

Purpose:

- tests multi-dimensional conditions;
- simple, deterministic, bounded, and easy to visualize with slices.

This benchmark is optional and should be added only after the 1D core campaign is stable.

## Optional extension F7 - Friedman-style nonlinear regression

Name: `friedman5d`

Domain:

`x1,...,x5 in [0,1]`, independently uniform real values.

Target:

`f(x) = 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5`

Purpose:

- standard nonlinear regression structure;
- exercises higher-dimensional local approximation.

Use only as a secondary extension. The core paper-derived 1D suite should be completed first.

## Output range and epsilon0

Do not use one absolute `epsilon0` blindly across functions with different target ranges.

Preferred campaign rule:

`epsilon0 = epsilon_fraction * (max(f) - min(f))`

with a fixed `epsilon_fraction` shared across implementations. Candidate values:

- primary: 0.05
- optional strict setting: 0.025

Compute the range analytically when trivial or from a very dense deterministic grid. Store the actual numeric epsilon0 in the resolved run configuration.

## Benchmark validation

Add unit tests that:

1. compare function outputs to known hand-calculated points;
2. confirm float input is preserved;
3. validate finite outputs across each domain;
4. validate the evaluation grid endpoints;
5. save a reference target-function plot before any XCSF experiments are run.
