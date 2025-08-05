# Project Name
TARGET = app
TEST_STORAGE = test_storage
UNIT_TESTS = unit_tests
TEST_ANNOY = test_annoy
TEST_ALL = test_all_engines

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

# Storage test specific files
STORAGE_SRCS = $(SRC_DIR)/storage.cpp test_storage.cpp
STORAGE_OBJS = $(OBJ_DIR)/storage.o $(OBJ_DIR)/test_storage.o

# Unit test specific files (all source files except main.cpp)
UNIT_TEST_SRCS = $(wildcard $(SRC_DIR)/*.cpp) unit_tests.cpp
UNIT_TEST_OBJS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(UNIT_TEST_SRCS)))

# Annoy test specific files
ANNOY_TEST_SRCS = $(wildcard $(SRC_DIR)/*.cpp) test_annoy.cpp
ANNOY_TEST_OBJS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(ANNOY_TEST_SRCS)))

# All engines test specific files
ALL_ENGINES_TEST_SRCS = $(wildcard $(SRC_DIR)/*.cpp) test_all_engines.cpp
ALL_ENGINES_TEST_OBJS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(ALL_ENGINES_TEST_SRCS)))

# Default rule
all: $(TARGET)

# Compile and link main app
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile and link storage test
$(TEST_STORAGE): $(STORAGE_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile and link unit tests
$(UNIT_TESTS): $(UNIT_TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile and link annoy tests
$(TEST_ANNOY): $(ANNOY_TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile and link all engines test
$(TEST_ALL): $(ALL_ENGINES_TEST_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/main.o: main.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_storage.o: test_storage.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_annoy.o: test_annoy.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_all_engines.o: test_all_engines.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/unit_tests.o: unit_tests.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build
clean:
	rm -rf $(OBJ_DIR) $(TARGET) $(TEST_STORAGE) $(UNIT_TESTS) $(TEST_ANNOY) $(TEST_ALL) test_data.jsonl

# Run storage test
test-storage: $(TEST_STORAGE)
	./$(TEST_STORAGE)

# Run unit tests
test-unit: $(UNIT_TESTS)
	./$(UNIT_TESTS)

# Run annoy tests
test-annoy: $(TEST_ANNOY)
	./$(TEST_ANNOY)

# Run all engines comparison
test-all: $(TEST_ALL)
	./$(TEST_ALL)

# Phony targets
.PHONY: all clean test-storage test-unit test-annoy test-all
