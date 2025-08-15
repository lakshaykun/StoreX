# StoreX

StoreX is a simple C++ project that demonstrates the use of a Makefile for building and testing.

## Directory Structure

```
StoreX/
├── bin/                # Compiled binaries
├── include/            # Header files
├── obj/                # Object files
├── src/                # Source files
├── test/               # Test files
├── Makefile            # Makefile for building the project
└── README.md           # Project documentation
```

## Build Instructions

To build the project, run the following command:

```bash
make
```

This will compile the source files and create the necessary binaries in the `bin/` directory.

## Testing

To run the tests, use the following command:

```bash
make test
```

This will compile the test files and execute the tests.

## Cleaning Up

To clean up the generated files, you can run:

```bash
make clean
```

This will remove the `bin/`, `obj/`, and any other generated files.