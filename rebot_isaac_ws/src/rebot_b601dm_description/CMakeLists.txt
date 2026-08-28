cmake_minimum_required(VERSION 3.16)
project(rebot_b601dm_description NONE)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf meshes
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # Model gates: absolute + cross-model FK, joint limits, mimic, nq == 6,
  # frame contract, mesh URIs. See test/test_kinematics.py for why absolute
  # golden poses are asserted alongside parity.
  ament_add_pytest_test(test_kinematics test/test_kinematics.py
    TIMEOUT 300
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

ament_package()
