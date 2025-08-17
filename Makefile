CXX = g++
CXXFLAGS = -Wall -Wextra -g -std=c++17 -Iinclude
LDFLAGS = -lsqlite3 

SRC_DIR = src
TEST_DIR = test
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
MAIN_OBJ = $(OBJ_DIR)/main.o
SRC_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES))
TEST_OBJ = $(OBJ_DIR)/unit_tests.o
ADVANCED_TEST_OBJ = $(OBJ_DIR)/advanced_tests.o

MAIN_TARGET = $(BIN_DIR)/main
TEST_TARGET = $(BIN_DIR)/tests
ADVANCED_TEST_TARGET = $(BIN_DIR)/advanced_tests

all: $(MAIN_TARGET) $(TEST_TARGET) $(ADVANCED_TEST_TARGET)

# Main executable
$(MAIN_TARGET): $(MAIN_OBJ) $(SRC_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)  # <-- Added $(LDFLAGS)

# Test executable
$(TEST_TARGET): $(TEST_OBJ) $(SRC_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)  # <-- Added $(LDFLAGS)

# Advanced test executable
$(ADVANCED_TEST_TARGET): $(ADVANCED_TEST_OBJ) $(SRC_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Compile main.cpp
$(OBJ_DIR)/main.o: main.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile unit_tests.cpp
$(OBJ_DIR)/unit_tests.o: $(TEST_DIR)/unit_tests.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile advanced_tests.cpp
$(OBJ_DIR)/advanced_tests.o: $(TEST_DIR)/advanced_tests.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile all src/*.cpp
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(INC_DIR)/*.hpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

run: $(MAIN_TARGET)
	./$(MAIN_TARGET)

.PHONY: all clean
