# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/alery/clion-2019.2.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/alery/clion-2019.2.2/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alery/CLionProjects/c_tensorflow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alery/CLionProjects/c_tensorflow/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/c_tensorflow.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/c_tensorflow.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/c_tensorflow.dir/flags.make

CMakeFiles/c_tensorflow.dir/project_exp/use_awa_pb.c.o: CMakeFiles/c_tensorflow.dir/flags.make
CMakeFiles/c_tensorflow.dir/project_exp/use_awa_pb.c.o: ../project_exp/use_awa_pb.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alery/CLionProjects/c_tensorflow/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/c_tensorflow.dir/project_exp/use_awa_pb.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/c_tensorflow.dir/project_exp/use_awa_pb.c.o   -c /home/alery/CLionProjects/c_tensorflow/project_exp/use_awa_pb.c

CMakeFiles/c_tensorflow.dir/project_exp/use_awa_pb.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/c_tensorflow.dir/project_exp/use_awa_pb.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/alery/CLionProjects/c_tensorflow/project_exp/use_awa_pb.c > CMakeFiles/c_tensorflow.dir/project_exp/use_awa_pb.c.i

CMakeFiles/c_tensorflow.dir/project_exp/use_awa_pb.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/c_tensorflow.dir/project_exp/use_awa_pb.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/alery/CLionProjects/c_tensorflow/project_exp/use_awa_pb.c -o CMakeFiles/c_tensorflow.dir/project_exp/use_awa_pb.c.s

# Object files for target c_tensorflow
c_tensorflow_OBJECTS = \
"CMakeFiles/c_tensorflow.dir/project_exp/use_awa_pb.c.o"

# External object files for target c_tensorflow
c_tensorflow_EXTERNAL_OBJECTS =

c_tensorflow: CMakeFiles/c_tensorflow.dir/project_exp/use_awa_pb.c.o
c_tensorflow: CMakeFiles/c_tensorflow.dir/build.make
c_tensorflow: CMakeFiles/c_tensorflow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alery/CLionProjects/c_tensorflow/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable c_tensorflow"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/c_tensorflow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/c_tensorflow.dir/build: c_tensorflow

.PHONY : CMakeFiles/c_tensorflow.dir/build

CMakeFiles/c_tensorflow.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/c_tensorflow.dir/cmake_clean.cmake
.PHONY : CMakeFiles/c_tensorflow.dir/clean

CMakeFiles/c_tensorflow.dir/depend:
	cd /home/alery/CLionProjects/c_tensorflow/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alery/CLionProjects/c_tensorflow /home/alery/CLionProjects/c_tensorflow /home/alery/CLionProjects/c_tensorflow/cmake-build-debug /home/alery/CLionProjects/c_tensorflow/cmake-build-debug /home/alery/CLionProjects/c_tensorflow/cmake-build-debug/CMakeFiles/c_tensorflow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/c_tensorflow.dir/depend

