# Applications must load the selected SDK runtime before Windows' System32
# fallback. PATH alone cannot enforce that DLL search order.
function(rns8_collect_hip_runtime_dlls selected_runtime result)
  if(NOT EXISTS "${selected_runtime}" OR IS_DIRECTORY "${selected_runtime}")
    message(FATAL_ERROR "Selected HIP runtime DLL is missing: ${selected_runtime}")
  endif()
  get_filename_component(_runtime_dir "${selected_runtime}" DIRECTORY)
  set(_files "${selected_runtime}")
  # COMGR is loaded dynamically during device discovery, so import-table
  # staging alone can leave a complete SDK reporting zero available devices.
  foreach(_name IN ITEMS rocm_kpack.dll amd_comgr.dll)
    if(EXISTS "${_runtime_dir}/${_name}" AND NOT IS_DIRECTORY "${_runtime_dir}/${_name}")
      list(APPEND _files "${_runtime_dir}/${_name}")
    endif()
  endforeach()
  set(${result} "${_files}" PARENT_SCOPE)
endfunction()

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
