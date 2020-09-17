.PHONY: all clean

CPP = g++ 
ALPHA_FLAGS = -Isrc/
MATPLOTLIBCPP_FLAGS = -I/usr/include/python3.8 -I/usr/lib/python3.8 -lpython3.8
NUMCPP_FLAGS = 
BOOST_FLAGS =
CPP_FLAGS = -std=c++11 -g -o

# create build directory
CREATE_DIR : $(shell mkdir -p ./build)

# compile /examples dir with options to compile single
%: $(CREATE_DIR) ./examples/%.cpp
	@# @echo compiling $@.cpp
	@$(CPP) $^ $(ALPHA_FLAGS) $(MATPLOTLIBCPP_FLAGS) $(NUMCPP_FLAGS) \
	$(BOOST_FLAGS) $(CPP_FLAGS) ./build/$@
	@echo executable ./build/$@ created. 


# or all files 
all: $(patsubst ./examples/%.cpp, %, $(wildcard ./examples/*.cpp))

clean: 
	rm -rf ./build
	@echo built binaries cleared.

