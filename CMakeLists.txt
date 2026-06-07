cmake_minimum_required(VERSION 3.8)
project(jetank_manipulation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Generate the GraspObject action interface.
# NOTE: rosidl_generate_interfaces internally installs generated Python bindings
# under the project name, so we must NOT call ament_python_install_package()
# separately with the same project name (CMP0002 duplicate target).
# Instead, install the source Python package files directly.
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/GraspObject.action"
  DEPENDENCIES std_msgs builtin_interfaces
)

# Install source Python package alongside the generated bindings.
ament_get_python_install_dir(PYTHON_INSTALL_DIR)
install(DIRECTORY jetank_manipulation/
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}"
  FILES_MATCHING
    PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)

# Install the grasp server executable: ros2 run jetank_manipulation grasp_server
install(PROGRAMS
  jetank_manipulation/grasp_server.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME grasp_server
)

# Install launch and config directories
foreach(_dir launch config)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_dir}")
    install(DIRECTORY ${_dir}
      DESTINATION share/${PROJECT_NAME}
    )
  endif()
endforeach()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_xmllint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_import test/test_import.py)
endif()

ament_package()
