project(source)
add_executable(source source.cpp source.report.md.in.txt)
target_link_libraries(source ${OpenCV_LIBS})

file(READ source.cpp cpp_source)
# prepare report md file
configure_file(source.report.md.in.txt ${CMAKE_CURRENT_BINARY_DIR}/source.report.md @ONLY)
add_custom_command(TARGET ${PROJECT_NAME}
  POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/source.report.md" "$<TARGET_FILE_DIR:${PROJECT_NAME}>")