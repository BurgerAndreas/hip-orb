#!/usr/bin/env bash
# Source this file from the repository root. Exported settings override .env.
hip_load_environment() {
  local hip_env_path="${HIP_ENV_FILE:-$PWD/.env}" hip_key hip_index
  local -a hip_saved_names=() hip_saved_values=()
  while IFS= read -r hip_key; do
    case "$hip_key" in
      HIP_*|HF_*|FAIRCHEM_*)
        hip_saved_names+=("$hip_key")
        hip_saved_values+=("${!hip_key}")
        ;;
    esac
  done < <(compgen -e)
  if [[ -f "$hip_env_path" ]]; then
    local hip_had_allexport=false
    case "$-" in *a*) hip_had_allexport=true ;; esac
    set -a
    source "$hip_env_path"
    if [[ "$hip_had_allexport" == false ]]; then set +a; fi
  elif [[ -n "${HIP_ENV_FILE:-}" ]]; then
    echo 'The configured HIP_ENV_FILE does not exist.' >&2
    return 1
  fi
  for hip_index in "${!hip_saved_names[@]}"; do
    export "${hip_saved_names[$hip_index]}=${hip_saved_values[$hip_index]}"
  done
}
hip_load_environment
unset -f hip_load_environment
