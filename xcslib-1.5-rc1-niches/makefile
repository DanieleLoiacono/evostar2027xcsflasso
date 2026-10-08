#
#	Generates different versions of XCS
#
#	bf:	standard XCS for Boolean functions 
#	woods:	standard XCS for Woods environment 
#

BUILD_DIR := ./build
EXECUTABLES=`find . -maxdepth 1 -type f -perm +a+x`
OBJECTS=$(wildcard ./*.o ./extra/*.o ./pf/*.o)
DOCS=$(wildcard ./docs/html/*.html ./docs/html/*.png)
BAK = $(wildcard ./*~)
EXECUTION=$(wildcard pop_state.* statistics.* *.gz popul.* steps.* report.* AVERAGE.* average.*)

clean:
	@\rm -f $(OBJECTS) $(BAK) 
	@\rm -rf $(BUILD_DIR)

clean-all:
	@\rm -f $(OBJECTS) $(BAK) $(EXECUTABLES) *.gz
	@\rm -f example/*/*.gz statistics.* trace.* steps.* perf.* popul.* report.* system_error.* AVERAGE.* average.*


.PHONY: docs clean clean-all

################################################################################
#
# TERNARY REPRESENTATION
#
################################################################################


### XCS for Boolean Functions
bf:
	make -f make/xcs.make ENVIRONMENT_VERSION=-bf ENVIRONMENT=bf_env USERFLAGS="-D__NICHE_TRACKING__"

### XCS for Woods Environments 
woods:
	make -f make/xcs.make ENVIRONMENT_VERSION=-woods ENVIRONMENT=woods_env ACTIONS=binary_action  USERFLAGS="-D__NICHE_TRACKING__"

### XCS for Real Valued Functions
rf:
	make clean; make -f make/xcsf.make VERSION="-rf" ACTIONS=dummy_action USERFLAGS="-D__NICHE_TRACKING__"

### XCS for Real Valued Grid Worlds
gw:
	make clean; make -f make/xcsf.make ENVIRONMENT=grid_env VERSION="-gw" ACTIONS=integer_action USERFLAGS="-D__NICHE_TRACKING__"
