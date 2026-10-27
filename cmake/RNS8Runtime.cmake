# Applications must load the selected SDK runtime before Windows' System32
# fallback. PATH alone cannot enforce that DLL search order.
function(rns8_copy_runtime_dependencies target)
  if(NOT WIN32 OR NOT RNS8_WINDOWS_RUNTIME_DLLS)
    return()
  endif()
  get_target_property(_attached ${target} RNS8_RUNTIME_STAGING_ATTACHED)
  if(_attached)
    return()
  endif()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND "${CMAKE_COMMAND}"
      "-DRNS8_RUNTIME_DLLS=${RNS8_WINDOWS_RUNTIME_DLLS}"
      "-DRNS8_RUNTIME_DESTINATION=$<TARGET_FILE_DIR:${target}>"
      -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/RNS8StageRuntime.cmake"
    VERBATIM COMMENT "Staging the selected RNS8 HIP runtime beside ${target}")
  set_property(TARGET ${target} PROPERTY RNS8_RUNTIME_STAGING_ATTACHED TRUE)
endfunction()
