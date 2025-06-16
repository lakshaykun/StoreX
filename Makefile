# Project Name
TARGET = app

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# Directories
INCLUDES = -Iinclude
SRC_DIR = src
OBJ_DIR = obj

# Source and object files
SRCS = $(wildcard $(SRC_DIR)/*.cpp) main.cpp
OBJS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(SRCS)))

# Default rule
all: $(TARGET)

# Compile and link
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/main.o: main.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

# Phony targets
.PHONY: all clean
