include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/MatrixTargetContract.cmake")

function(rns8_rocwmma_target_contract output)
  rns8_matrix_target_contract(_wave rocWMMA ${ARGN})
  set(${output} ${_wave} PARENT_SCOPE)
endfunction()
