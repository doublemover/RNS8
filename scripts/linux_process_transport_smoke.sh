#!/usr/bin/env bash
set -euo pipefail
if (( $# < 4 || $# > 5 )); then
  echo "usage: $0 binary mpi-host|rccl|rocshmem modulus|rows|columns|contraction|hybrid rank-local-ordinals [hostfile]" >&2
  exit 2
fi
binary=$1
transport=$2
axis=$3
devices=$4
[[ $transport =~ ^(mpi-host|rccl|rocshmem)$ && $axis =~ ^(modulus|rows|columns|contraction|hybrid)$ && $devices =~ ^[0-9]+(,[0-9]+)*$ ]] || exit 2
IFS=, read -r -a ordinals <<< "$devices"
(( ${#ordinals[@]} >= 1 && ${#ordinals[@]} <= 64 )) || exit 2
args=(-np "${#ordinals[@]}")
if (( $# == 5 )); then args+=(--hostfile "$5"); fi
# Bounds MPI/rocSHMEM setup/teardown as well as the session's own progress waits.
# Never change system clocks, peer policy, partition modes or MPI fabric options.
exec timeout --signal=TERM --kill-after=10s 300s mpiexec "${args[@]}" "$binary" "$transport" "$axis" "$devices"
