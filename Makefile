# Project Name
TARGET = app
TEST_STORAGE = test_storage

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

# Default rule
all: $(TARGET)

# Compile and link main app
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile and link storage test
$(TEST_STORAGE): $(STORAGE_OBJS)
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

# Clean build
clean:
	rm -rf $(OBJ_DIR) $(TARGET) $(TEST_STORAGE) test_data.jsonl

# Run storage test
test-storage: $(TEST_STORAGE)
	./$(TEST_STORAGE)

# Phony targets
.PHONY: all clean test-storage
