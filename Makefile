CFLAGS = -O -std=c++17
CC = g++
LIB = -lmlpack -larmadillo -fopenmp
linreg: main.o load.o
	$(CC) $(CFLAGS) -o linreg main.o load.o $(LIB)
loadexec: load.o
	$(CC) $(CFLAGS) -o load load.o $(LIB)
main.o: main.cc
	$(CC) $(CFLAGS) -c main.cc
load.o: load.cc
	$(CC) $(CFLAGS) -c load.cc
