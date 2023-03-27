CXXFLAGS += -I. $(shell root-config --cflags) -g
LDFLAGS += $(shell root-config --libs) -lPhysics -lMatrix -g

PROGRAMS = WaveformStudy 

all:	clean $(PROGRAMS)

$(PROGRAMS):
	@echo '<<compiling' $@'>>'
	#@$(CXX) $@.cpp *.o -o $@ $(CXXFLAGS) $(LDFLAGS)
	@$(CXX) $@.cpp -o $@ $(CXXFLAGS) $(LDFLAGS)
	@rm -rf *.dSYM
clean:	
	rm -f $(PROGRAMS)
