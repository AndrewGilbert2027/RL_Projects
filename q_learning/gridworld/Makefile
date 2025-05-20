CXX = g++
CXXFLAGS = -Wall -g
SRC = train.c++ grid.c++
OBJ = $(SRC:.c++=.o)
TARGET = train

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.c++
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
# End of Makefile