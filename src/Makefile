# this makefile is intended for g++ on Linux

CC = g++
CFLAGS = -c -Wall -O3 -std=c++17
LDFLAGS = -pthread -lncurses
INCLUDE = -I../include/
SOURCES = project.cpp iofunc.cpp filter.cpp  fourier.cpp  genfunc.cpp  logfunc.cpp demod.cpp pll.cpp rds.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = project

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDE) $< -o $@

clean:
	-rm $(OBJECTS)
