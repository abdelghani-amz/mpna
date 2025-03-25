HYPRE_DIR = $(HOME)/hypre/src/hypre
CXX = mpicxx
CXXFLAGS = -O2 -I$(HYPRE_DIR)/include
LDFLAGS = -L$(HYPRE_DIR)/lib -lHYPRE

all: 
	$(CXX) $(CXXFLAGS) nonLinear.cpp -o main $(LDFLAGS) 
