CC=gcc
CFLAGS= -fopenmp -g -Wall -O2 -std=gnu99
LDFLAGS= -fopenmp

SRC := $(wildcard *.c)
SRC := $(filter-out cpu_stats.c, $(SRC))
TARGET=$(SRC:.c=.run)

all: $(TARGET)

%.run: %.o
	$(CC) -o $@ $^ cpu_stats.o $(LDFLAGS)

%.o: %.c cpu_stats.c
	$(CC) -c $^ cpu_stats.c $(CFLAGS)

clean:
	rm -rf *.o *.run

.PHONY: clean
