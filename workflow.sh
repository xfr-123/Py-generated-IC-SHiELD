#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$REPO_ROOT/$(basename "${BASH_SOURCE[0]}")"
NOTEBOOK_DIR="$REPO_ROOT/publication_notebooks"

PYGEN_PYTHON="${PYGEN_PYTHON:-python}"
JUPYTER_PYTHON="${JUPYTER_PYTHON:-python}"
SHIELD_RUNTIME="${SHIELD_RUNTIME:-singularity}"
SHIELD_EXECUTABLE="${SHIELD_EXECUTABLE:-/SHiELD_build/Build/bin/SOLO_nh.prod.32bit.gnu.x}"
SHIELD_NTASKS="${SHIELD_NTASKS:-24}"
SHIELD_CPUS_PER_TASK="${SHIELD_CPUS_PER_TASK:-1}"
SHIELD_NODES="${SHIELD_NODES:-2}"
SHIELD_TASKS_PER_NODE="${SHIELD_TASKS_PER_NODE:-12}"
SHIELD_TIME="${SHIELD_TIME:-02:00:00}"
CASE_TIME="${CASE_TIME:-05:00:00}"
POST_RUNTIME="${POST_RUNTIME:-$SHIELD_RUNTIME}"
POST_BINDIR="${POST_BINDIR:-/opt/ufs-srweather-app/container-bin}"
POST_TIME="${POST_TIME:-01:00:00}"
POST_NLON="${POST_NLON:-360}"
POST_NLAT="${POST_NLAT:-180}"
PUBLICATION_NOTEBOOK_TIMEOUT="${PUBLICATION_NOTEBOOK_TIMEOUT:-0}"

log() {
    printf '[workflow] %s\n' "$*"
}

warn() {
    printf '[workflow] warning: %s\n' "$*" >&2
}

die() {
    printf '[workflow] error: %s\n' "$*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || die "required file is missing: ${1#$REPO_ROOT/}"
}

require_dir() {
    [[ -d "$1" ]] || die "required directory is missing: $1"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command is not available: $1"
}

run_in_repo() {
    (
        cd "$REPO_ROOT"
        "$@"
    )
}

resolve_directory() {
    require_dir "$1"
    (cd "$1" && pwd -P)
}

usage() {
    cat <<'EOF'
Reproducible SHiELD simulation and publication-figure workflow

Usage:
  ./workflow.sh COMMAND [OPTIONS]

Simulation commands:
  check [--data-root DIR]           Check software, scripts, and optional paper data.
  prepare-cold                      Create the cold-start model configuration.
  run-model [--submit] [--wait]     Run SHiELD now or submit it through Slurm.
  build-weights                     Build C96/latitude-longitude remapping weights.
  generate-ic [--] IC_OPTIONS       Generate restart fields with ic_generator.py.
  install-restart                   Install generated CACHE fields into RESTART.
  prepare-warm moist|dry            Create the requested warm-start configuration.
  postprocess [--submit] [--wait]   Remap SHiELD tile output to latitude-longitude.
  case [OPTIONS] [-- IC_OPTIONS]    Run the complete cold-to-warm experiment.

Notebook commands:
  list-notebooks                    List the publication notebooks.
  notebooks --data-root DIR         Open the paper-data notebook workspace in Jupyter.
  figures --data-root DIR [OPTIONS] Execute publication notebooks and rebuild figures.

Complete-case options:
  --mode moist|dry                  Warm-start mode (default: moist).
  --submit                          Submit the complete workflow through Slurm.
  --wait                            Wait for a submitted Slurm job to finish.
  --no-postprocess                  Stop after the warm SHiELD integration.

Figure options:
  --notebook NAME                   Execute one notebook; may be repeated.
  --output-dir DIR                  Executed-notebook directory, relative to data root
                                    unless absolute (default: publication_notebooks/executed).
  --timeout SECONDS                 Per-cell timeout; 0 disables it (default: 0).

Required environment variables for model execution:
  SHIELD_CONTAINER                  SHiELD Singularity/Apptainer image.
  POST_CONTAINER                    Postprocessing image containing fregrid tools.

Useful environment overrides:
  PYGEN_PYTHON, JUPYTER_PYTHON, SHIELD_RUNTIME, POST_RUNTIME
  SHIELD_EXECUTABLE, SHIELD_NTASKS, SHIELD_CPUS_PER_TASK
  SHIELD_PARTITION, SHIELD_ACCOUNT, SHIELD_TIME, CASE_TIME
  POST_BINDIR, POST_PARTITION, POST_TIME, POST_NLON, POST_NLAT
  PUBLICATION_DATA_ROOT, PUBLICATION_NOTEBOOK_TIMEOUT

Examples:
  ./workflow.sh check --data-root /path/to/paper-data
  ./workflow.sh prepare-cold
  ./workflow.sh run-model --submit --wait
  ./workflow.sh build-weights
  ./workflow.sh generate-ic -- --IsPerturbation --Shift 10 --b 2 --n 3 --RH0 0.8
  ./workflow.sh install-restart
  ./workflow.sh prepare-warm moist
  ./workflow.sh case --mode dry --submit -- --Shift 10 --b 2 --n 3 --RH0 0
  ./workflow.sh figures --data-root /path/to/paper-data
  ./workflow.sh figures --data-root /path/to/paper-data --notebook 01_main_figures_02_06.ipynb

The publication data archive is intentionally kept outside this code repository. Its root
must contain ready_version/, derived_data/, publication_notebooks/figure_code_inventory.csv,
the BCwave_*.nc files, and the analysis directories used by the selected notebook. Executed
notebooks and generated figures are written under that data root, so the clean source
notebooks in this repository are not overwritten.
EOF
}

check_data_root() {
    local data_root="$1"
    local missing=0
    local required

    for required in \
        ready_version \
        derived_data \
        publication_notebooks/figure_code_inventory.csv; do
        if [[ ! -e "$data_root/$required" ]]; then
            warn "paper-data item is missing: $required"
            missing=1
        fi
    done

    if ! compgen -G "$data_root/BCwave_*.nc" >/dev/null; then
        warn 'paper-data root contains no BCwave_*.nc files'
        missing=1
    fi

    ((missing == 0)) || return 1
}

check_environment() {
    local data_root=""

    while (($#)); do
        case "$1" in
            --data-root)
                (($# >= 2)) || die '--data-root requires a directory'
                data_root="$2"
                shift 2
                ;;
            -h|--help)
                usage
                return
                ;;
            *)
                die "unknown check option: $1"
                ;;
        esac
    done

    local required_scripts=(
        prep_cold.sh
        modify_restart.sh
        ic_generator.py
        prep_warm.sh
        prep_warm_dry.sh
    )
    local script
    for script in "${required_scripts[@]}"; do
        require_file "$REPO_ROOT/$script"
    done
    require_dir "$NOTEBOOK_DIR"
    log 'repository scripts and publication notebooks are present'

    require_command "$PYGEN_PYTHON"
    "$PYGEN_PYTHON" -c 'import numpy, xarray' >/dev/null \
        || die "$PYGEN_PYTHON cannot import the IC-generation dependencies"
    log 'IC-generation Python environment is available'

    if command -v gridspec-create >/dev/null 2>&1 \
        && command -v ESMF_RegridWeightGen >/dev/null 2>&1; then
        log 'ESMF remapping commands are available'
    else
        warn 'activate the esmpy environment before running build-weights'
    fi

    if command -v "$SHIELD_RUNTIME" >/dev/null 2>&1; then
        log "container runtime is available: $SHIELD_RUNTIME"
    else
        warn "container runtime is unavailable: $SHIELD_RUNTIME"
    fi

    [[ -n "${SHIELD_CONTAINER:-}" ]] \
        || warn 'set SHIELD_CONTAINER before running the model'
    [[ -n "${POST_CONTAINER:-}" ]] \
        || warn 'set POST_CONTAINER before postprocessing'

    MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/py-generated-ic-shield-matplotlib}" \
    "$JUPYTER_PYTHON" -c \
        'import IPython, matplotlib, nbclient, nbformat, numpy, pandas, scipy, xarray' \
        >/dev/null \
        || die "$JUPYTER_PYTHON cannot import the publication-notebook dependencies"
    log 'publication-notebook Python environment is available'

    if [[ -n "$data_root" ]]; then
        data_root="$(resolve_directory "$data_root")"
        check_data_root "$data_root" \
            || die 'paper-data archive is incomplete; see warnings above'
        log 'paper-data archive has the common required inputs'
    fi
}

prepare_cold() {
    require_file "$REPO_ROOT/prep_cold.sh"
    log 'preparing cold-start configuration'
    run_in_repo bash ./prep_cold.sh
}

build_weights() {
    require_file "$REPO_ROOT/modify_restart.sh"
    require_command gridspec-create
    require_command ESMF_RegridWeightGen
    log 'building C96 remapping weights in CACHE/'
    run_in_repo bash ./modify_restart.sh
}

generate_ic() {
    require_file "$REPO_ROOT/ic_generator.py"
    require_dir "$REPO_ROOT/RESTART"
    require_dir "$REPO_ROOT/CACHE"
    require_command "$PYGEN_PYTHON"

    if [[ "${1:-}" == '--' ]]; then
        shift
    fi

    log 'generating Python-defined restart fields in CACHE/'
    run_in_repo "$PYGEN_PYTHON" ./ic_generator.py "$@"
}

install_restart() {
    require_dir "$REPO_ROOT/RESTART"
    require_dir "$REPO_ROOT/CACHE"

    local generated_core generated_tracer
    generated_core=("$REPO_ROOT"/CACHE/fv_core.res.tile*)
    generated_tracer=("$REPO_ROOT"/CACHE/fv_tracer.res.tile*)

    [[ -e "${generated_core[0]}" ]] || die 'CACHE/fv_core.res.tile* was not generated'
    [[ -e "${generated_tracer[0]}" ]] || die 'CACHE/fv_tracer.res.tile* was not generated'

    log 'replacing RESTART core and tracer tiles with generated fields'
    rm -f "$REPO_ROOT"/RESTART/fv_core.res.tile* \
        "$REPO_ROOT"/RESTART/fv_tracer.res.tile*
    cp "${generated_core[@]}" "${generated_tracer[@]}" "$REPO_ROOT/RESTART/"
}

prepare_warm() {
    local mode="${1:-}"

    case "$mode" in
        moist)
            require_file "$REPO_ROOT/prep_warm.sh"
            log 'preparing moist warm-start configuration'
            run_in_repo bash ./prep_warm.sh
            ;;
        dry)
            require_file "$REPO_ROOT/prep_warm_dry.sh"
            log 'preparing dry warm-start configuration'
            run_in_repo bash ./prep_warm_dry.sh
            ;;
        *)
            die 'prepare-warm requires mode "moist" or "dry"'
            ;;
    esac
}

run_model_now() {
    [[ -n "${SHIELD_CONTAINER:-}" ]] || die 'SHIELD_CONTAINER is not set'
    require_file "$SHIELD_CONTAINER"
    require_command "$SHIELD_RUNTIME"

    local command=(
        "$SHIELD_RUNTIME" exec --no-home
        "$SHIELD_CONTAINER"
        "$SHIELD_EXECUTABLE"
    )

    log 'running SHiELD'
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        require_command srun
        run_in_repo srun --ntasks="$SHIELD_NTASKS" \
            --cpus-per-task="$SHIELD_CPUS_PER_TASK" "${command[@]}"
    else
        warn 'no Slurm allocation detected; running the container directly'
        run_in_repo "${command[@]}"
    fi
}

run_post_task() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        srun --ntasks=1 --cpus-per-task=1 "$@"
    else
        "$@"
    fi
}

postprocess_now() {
    [[ -n "${POST_CONTAINER:-}" ]] || die 'POST_CONTAINER is not set'
    require_file "$POST_CONTAINER"
    require_command "$POST_RUNTIME"
    [[ -z "${SLURM_JOB_ID:-}" ]] || require_command srun

    local container=("$POST_RUNTIME" exec --no-home "$POST_CONTAINER")
    local make_hgrid="$POST_BINDIR/make_hgrid"
    local make_solo_mosaic="$POST_BINDIR/make_solo_mosaic"
    local fregrid="$POST_BINDIR/fregrid"
    local daily_fields="${POST_DAILY_FIELDS:-PWAT,h_plev,u_plev,v_plev,t_plev,PRESsfc,VORT850,VORT500,VORT200,omg500}"
    local hourly_fields="${POST_HOURLY_FIELDS:-PWAT,h_plev,u_plev,v_plev,t_plev,q_plev,omg_plev,PRESsfc}"

    log 'creating the cubed-sphere mosaic and latitude-longitude products'
    (
        cd "$REPO_ROOT"
        run_post_task "${container[@]}" "$make_hgrid" \
            --grid_type gnomonic_ed --nlon 192
        run_post_task "${container[@]}" "$make_solo_mosaic" \
            --num_tiles 6 --dir . \
            --tile_file horizontal_grid.tile1.nc,horizontal_grid.tile2.nc,horizontal_grid.tile3.nc,horizontal_grid.tile4.nc,horizontal_grid.tile5.nc,horizontal_grid.tile6.nc
        run_post_task "${container[@]}" "$fregrid" \
            --input_mosaic solo_mosaic.nc --nlon "$POST_NLON" --nlat "$POST_NLAT" \
            --input_file atmos_daily --scalar_field "$daily_fields"
        run_post_task "${container[@]}" "$fregrid" \
            --input_mosaic solo_mosaic.nc --nlon "$POST_NLON" --nlat "$POST_NLAT" \
            --input_file atmos_4x_hourly --scalar_field "$hourly_fields"
    )
}

submit_self() {
    local profile="$1"
    local wait_for_job="$2"
    shift 2

    require_command sbatch

    local options=(--parsable --export=ALL)
    local partition account walltime job_name nodes tasks_per_node cpus_per_task

    if [[ "$profile" == post ]]; then
        partition="${POST_PARTITION:-}"
        account="${POST_ACCOUNT:-${SHIELD_ACCOUNT:-}}"
        walltime="$POST_TIME"
        job_name='SHIELD_post'
        nodes=1
        tasks_per_node=1
        cpus_per_task=1
    elif [[ "$profile" == case ]]; then
        partition="${SHIELD_PARTITION:-}"
        account="${SHIELD_ACCOUNT:-}"
        walltime="$CASE_TIME"
        job_name='SHIELD_case'
        nodes="$SHIELD_NODES"
        tasks_per_node="$SHIELD_TASKS_PER_NODE"
        cpus_per_task="$SHIELD_CPUS_PER_TASK"
    else
        partition="${SHIELD_PARTITION:-}"
        account="${SHIELD_ACCOUNT:-}"
        walltime="$SHIELD_TIME"
        job_name='SHIELD_run'
        nodes="$SHIELD_NODES"
        tasks_per_node="$SHIELD_TASKS_PER_NODE"
        cpus_per_task="$SHIELD_CPUS_PER_TASK"
    fi

    options+=(
        --time="$walltime"
        --nodes="$nodes"
        --ntasks-per-node="$tasks_per_node"
        --cpus-per-task="$cpus_per_task"
        --job-name="$job_name"
        --output="${job_name}.o%j"
    )
    [[ -z "$partition" ]] || options+=(--partition="$partition")
    [[ -z "$account" ]] || options+=(--account="$account")
    [[ "$wait_for_job" == true ]] && options+=(--wait)

    log "submitting $job_name through Slurm"
    (cd "$REPO_ROOT" && sbatch "${options[@]}" "$SCRIPT_PATH" "$@")
}

run_model() {
    local submit=false
    local wait_for_job=false

    while (($#)); do
        case "$1" in
            --submit) submit=true ;;
            --wait) wait_for_job=true ;;
            -h|--help)
                usage
                return
                ;;
            *) die "unknown run-model option: $1" ;;
        esac
        shift
    done

    [[ "$wait_for_job" == false || "$submit" == true ]] \
        || die '--wait requires --submit'

    if [[ "$submit" == true ]]; then
        submit_self model "$wait_for_job" __run-model
    else
        run_model_now
    fi
}

postprocess() {
    local submit=false
    local wait_for_job=false

    while (($#)); do
        case "$1" in
            --submit) submit=true ;;
            --wait) wait_for_job=true ;;
            -h|--help)
                usage
                return
                ;;
            *) die "unknown postprocess option: $1" ;;
        esac
        shift
    done

    [[ "$wait_for_job" == false || "$submit" == true ]] \
        || die '--wait requires --submit'

    if [[ "$submit" == true ]]; then
        submit_self post "$wait_for_job" __postprocess
    else
        postprocess_now
    fi
}

run_case_now() {
    local mode="$1"
    local do_postprocess="$2"
    shift 2

    prepare_cold
    run_model_now
    build_weights
    generate_ic "$@"
    install_restart
    prepare_warm "$mode"
    run_model_now
    if [[ "$do_postprocess" == true ]]; then
        postprocess_now
    fi
}

run_case() {
    local mode=moist
    local submit=false
    local wait_for_job=false
    local do_postprocess=true
    local ic_arguments=()

    while (($#)); do
        case "$1" in
            --mode)
                (($# >= 2)) || die '--mode requires moist or dry'
                mode="$2"
                shift 2
                ;;
            --submit)
                submit=true
                shift
                ;;
            --wait)
                wait_for_job=true
                shift
                ;;
            --no-postprocess)
                do_postprocess=false
                shift
                ;;
            --)
                shift
                ic_arguments=("$@")
                break
                ;;
            -h|--help)
                usage
                return
                ;;
            *)
                die "unknown case option: $1 (put ic_generator.py options after --)"
                ;;
        esac
    done

    [[ "$mode" == moist || "$mode" == dry ]] \
        || die '--mode must be moist or dry'
    [[ "$wait_for_job" == false || "$submit" == true ]] \
        || die '--wait requires --submit'

    local internal=(__case --mode "$mode")
    [[ "$do_postprocess" == true ]] || internal+=(--no-postprocess)
    internal+=(-- "${ic_arguments[@]}")

    if [[ "$submit" == true ]]; then
        submit_self case "$wait_for_job" "${internal[@]}"
    else
        run_case_now "$mode" "$do_postprocess" "${ic_arguments[@]}"
    fi
}

parse_internal_case() {
    local mode=moist
    local do_postprocess=true

    while (($#)); do
        case "$1" in
            --mode)
                mode="$2"
                shift 2
                ;;
            --no-postprocess)
                do_postprocess=false
                shift
                ;;
            --)
                shift
                break
                ;;
            *) die "invalid internal case argument: $1" ;;
        esac
    done

    run_case_now "$mode" "$do_postprocess" "$@"
}

list_notebooks() {
    require_dir "$NOTEBOOK_DIR"
    local notebook
    for notebook in "$NOTEBOOK_DIR"/*.ipynb; do
        [[ -e "$notebook" ]] || continue
        printf '%s\n' "$(basename "$notebook")"
    done
}

parse_data_root() {
    local value="${PUBLICATION_DATA_ROOT:-}"
    while (($#)); do
        case "$1" in
            --data-root)
                (($# >= 2)) || die '--data-root requires a directory'
                value="$2"
                shift 2
                ;;
            *) die "unknown notebook-workspace option: $1" ;;
        esac
    done
    [[ -n "$value" ]] || die 'provide --data-root DIR or set PUBLICATION_DATA_ROOT'
    resolve_directory "$value"
}

open_notebooks() {
    local data_root
    data_root="$(parse_data_root "$@")"
    check_data_root "$data_root" \
        || die 'paper-data archive is incomplete; see warnings above'
    require_dir "$data_root/publication_notebooks"
    require_command "$JUPYTER_PYTHON"

    log 'opening the publication notebook workspace'
    (
        cd "$data_root"
        "$JUPYTER_PYTHON" -m jupyter lab publication_notebooks
    )
}

resolve_notebook() {
    local name="$1"
    [[ "$name" != */* ]] || die '--notebook accepts a notebook basename only'
    [[ "$name" == *.ipynb ]] || name="$name.ipynb"
    require_file "$NOTEBOOK_DIR/$name"
    printf '%s\n' "$NOTEBOOK_DIR/$name"
}

execute_notebook() {
    local source_notebook="$1"
    local data_root="$2"
    local output_notebook="$3"
    local timeout="$4"
    local runtime_root="${TMPDIR:-/tmp}/py-generated-ic-shield-jupyter"

    mkdir -p "$runtime_root/ipython" "$runtime_root/runtime" "$runtime_root/matplotlib"

    log "executing $(basename "$source_notebook")"
    PUBLICATION_REBUILD_FIGURES=1 \
    IPYTHONDIR="${IPYTHONDIR:-$runtime_root/ipython}" \
    JUPYTER_RUNTIME_DIR="${JUPYTER_RUNTIME_DIR:-$runtime_root/runtime}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-$runtime_root/matplotlib}" \
    "$JUPYTER_PYTHON" - \
        "$source_notebook" "$data_root" "$output_notebook" "$timeout" "$REPO_ROOT" <<'PY'
from pathlib import Path
import os
import sys

import nbformat
from nbclient import NotebookClient

source = Path(sys.argv[1]).resolve()
data_root = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
timeout_value = int(sys.argv[4])
code_root = Path(sys.argv[5]).resolve()

notebook = nbformat.read(source, as_version=4)
client = NotebookClient(
    notebook,
    timeout=None if timeout_value == 0 else timeout_value,
    kernel_name=os.environ.get("PUBLICATION_KERNEL", "python3"),
    resources={"metadata": {"path": str(data_root)}},
    allow_errors=False,
    record_timing=False,
)
client.execute()

replacements = {
    str(data_root): ".",
    str(code_root): "<code-repository>",
}

def clean_text(value):
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [clean_text(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_text(item) for key, item in value.items()}
    return value

for cell in notebook.cells:
    for item in cell.get("outputs", []):
        for key in ("text", "traceback", "ename", "evalue"):
            if key in item:
                item[key] = clean_text(item[key])
        data = item.get("data", {})
        for mime_type in ("text/plain", "text/html", "application/json"):
            if mime_type in data:
                data[mime_type] = clean_text(data[mime_type])

output.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, output)
PY
}

rebuild_figures() {
    local data_root="${PUBLICATION_DATA_ROOT:-}"
    local output_dir=""
    local timeout="$PUBLICATION_NOTEBOOK_TIMEOUT"
    local selected=()

    while (($#)); do
        case "$1" in
            --data-root)
                (($# >= 2)) || die '--data-root requires a directory'
                data_root="$2"
                shift 2
                ;;
            --notebook)
                (($# >= 2)) || die '--notebook requires a name'
                selected+=("$(resolve_notebook "$2")")
                shift 2
                ;;
            --output-dir)
                (($# >= 2)) || die '--output-dir requires a directory'
                output_dir="$2"
                shift 2
                ;;
            --timeout)
                (($# >= 2)) || die '--timeout requires seconds'
                timeout="$2"
                shift 2
                ;;
            -h|--help)
                usage
                return
                ;;
            *) die "unknown figures option: $1" ;;
        esac
    done

    [[ -n "$data_root" ]] \
        || die 'provide --data-root DIR or set PUBLICATION_DATA_ROOT'
    [[ "$timeout" =~ ^[0-9]+$ ]] || die '--timeout must be a non-negative integer'
    data_root="$(resolve_directory "$data_root")"
    check_data_root "$data_root" \
        || die 'paper-data archive is incomplete; see warnings above'
    require_command "$JUPYTER_PYTHON"
    "$JUPYTER_PYTHON" -c 'import nbclient, nbformat' >/dev/null \
        || die "$JUPYTER_PYTHON cannot import nbclient and nbformat"

    if ((${#selected[@]} == 0)); then
        selected=("$NOTEBOOK_DIR"/0[1-9]_*.ipynb)
        [[ -e "${selected[0]}" ]] || die 'no figure notebooks were found'
    fi

    if [[ -z "$output_dir" ]]; then
        output_dir="$data_root/publication_notebooks/executed"
    elif [[ "$output_dir" != /* ]]; then
        output_dir="$data_root/$output_dir"
    fi
    mkdir -p "$output_dir"

    warn 'full figure rebuilding can require substantial memory, storage, and runtime'
    local notebook
    for notebook in "${selected[@]}"; do
        execute_notebook "$notebook" "$data_root" \
            "$output_dir/$(basename "$notebook")" "$timeout"
    done

    log 'publication figures were rebuilt under publication_notebooks/outputs/'
    log 'executed notebook copies were saved separately and source notebooks were unchanged'
}

main() {
    local command="${1:-help}"
    (($# == 0)) || shift

    case "$command" in
        help|-h|--help) usage ;;
        check) check_environment "$@" ;;
        prepare-cold) prepare_cold "$@" ;;
        run-model) run_model "$@" ;;
        build-weights) build_weights "$@" ;;
        generate-ic) generate_ic "$@" ;;
        install-restart) install_restart "$@" ;;
        prepare-warm) prepare_warm "$@" ;;
        postprocess) postprocess "$@" ;;
        case) run_case "$@" ;;
        list-notebooks) list_notebooks "$@" ;;
        notebooks) open_notebooks "$@" ;;
        figures) rebuild_figures "$@" ;;
        __run-model) run_model_now ;;
        __postprocess) postprocess_now ;;
        __case) parse_internal_case "$@" ;;
        *)
            usage >&2
            die "unknown command: $command"
            ;;
    esac
}

main "$@"
