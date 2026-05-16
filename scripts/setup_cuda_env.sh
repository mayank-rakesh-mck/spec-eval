#!/usr/bin/env bash
# scripts/setup_cuda_env.sh
#
# One-shot, idempotent CUDA toolchain setup for the spec-eval pipeline on a
# fresh Linux GPU box. Encodes every fix we hit getting sglang + flashinfer
# JIT to compile against a conda-provided CUDA. Safe to re-run.
#
# What it does:
#   1.  Install Miniconda (correct arch: x86_64 or aarch64).
#   2.  Accept Anaconda Terms of Service (required on new installs).
#   3.  Install CUDA dev packages from the nvidia channel, individually
#       (the cuda-toolkit metapackage trips ClobberError on some envs).
#       Headers covered: cudart, nvcc, nvrtc, cccl, curand, cublas, cusparse,
#       cusolver, driver. These are everything sglang's flashinfer JIT
#       (cutlass_utils.cuh -> tensor_fill.h -> curand_kernel.h, etc.) needs.
#   4.  Symlink headers AND libs from targets/<arch>/{include,lib}/ into
#       $CONDA_PREFIX/{include,lib} (conda's libcurand-dev / libcublas-dev /
#       libcusparse-dev / libcusolver-dev / cuda-nvrtc-dev ship under targets/,
#       but flashinfer's nvcc only -isystem's $CONDA_PREFIX/include and the
#       linker only -L's $CONDA_PREFIX/lib).
#   5.  Extra lib symlinks: lib64 -> lib (build systems expect lib64),
#       libcudart.so unversioned, libcuda.so -> driver stub.
#   6.  Generate ~/.spec-eval-cuda.env (sourceable). Sets CUDA_HOME, PATH,
#       LD_LIBRARY_PATH, CFLAGS/CXXFLAGS/LDFLAGS, NVCC_PREPEND_FLAGS
#       (-allow-unsupported-compiler so newer system gcc doesn't trip nvcc),
#       TORCH_CUDA_ARCH_LIST.
#   7.  Optionally reinstall torch from the matching cuXXX wheel index, and
#       optionally (re)install sglang[all].
#   8.  Optionally flush ~/.cache/flashinfer so stale JIT artifacts are rebuilt.
#
# Override behavior via env vars (see CONFIG block).
# Run `scripts/setup_cuda_env.sh --help` for flags.

set -euo pipefail

#==============================================================================
# CONFIG (override via env vars on the command line)
#==============================================================================
: "${CUDA_VERSION:=12.4}"
: "${MINICONDA_PREFIX:=$HOME/miniconda3}"
: "${TORCH_CUDA_TAG:=cu124}"          # PyTorch wheel index tag — must match CUDA_VERSION
: "${TORCH_VERSION:=2.4.*}"           # empty/star = latest in series
: "${ENV_FILE:=$HOME/.spec-eval-cuda.env}"
: "${TORCH_CUDA_ARCH_LIST:=8.0;8.6;8.9;9.0+PTX}"
: "${INSTALL_GCC13:=0}"               # 1 = install gcc 13 via conda (alternative to -allow-unsupported-compiler)

# Step toggles (1 = run, 0 = skip)
: "${DO_MINICONDA:=1}"
: "${DO_TOS:=1}"
: "${DO_CUDA_PKGS:=1}"
: "${DO_SYMLINKS:=1}"
: "${DO_ENV_FILE:=1}"
: "${DO_TORCH:=0}"
: "${DO_SGLANG:=0}"
: "${DO_FLUSH_CACHES:=0}"
: "${DO_DOCTOR:=1}"

# Verbose symlink reporting (1 = log each link). Default off to avoid noise on re-runs.
: "${SYMLINK_VERBOSE:=0}"

#==============================================================================
# LOGGING
#==============================================================================
_color() { [ -t 1 ] && printf '\033[%sm' "$1" || true; }
log()  { printf '%s[setup-cuda]%s %s\n' "$(_color '1;36')" "$(_color 0)" "$*"; }
warn() { printf '%s[setup-cuda]%s %s\n' "$(_color '1;33')" "$(_color 0)" "$*" >&2; }
err()  { printf '%s[setup-cuda]%s %s\n' "$(_color '1;31')" "$(_color 0)" "$*" >&2; }
ok()   { printf '%s[setup-cuda]%s %s\n' "$(_color '1;32')" "$(_color 0)" "$*"; }

#==============================================================================
# HELP
#==============================================================================
usage() {
    cat <<EOF
Usage: $(basename "$0") [flags]

Idempotent one-shot CUDA toolchain setup for spec-eval.

Flags:
  --with-torch          Also (re)install torch==${TORCH_VERSION} from ${TORCH_CUDA_TAG}.
  --with-sglang         Also (re)install sglang[all].
  --flush-caches        Clear ~/.cache/flashinfer (forces a clean JIT rebuild).
  --skip-doctor         Don't run final verification.
  --doctor-only         Run verification only (no install).
  --symlinks-only       Re-run header/lib symlinks + doctor (use after a new
                        'conda install -c nvidia libxxx-dev' when flashinfer
                        is missing a header).
  --install-gcc13       Install gcc 13 via conda (alternative to -allow-unsupported-compiler).
  --verbose-symlinks    Log every header/lib symlink as it is created.
  -h | --help           Show this message.

Env overrides (any can be set on the command line):
  CUDA_VERSION=${CUDA_VERSION}
  MINICONDA_PREFIX=${MINICONDA_PREFIX}
  TORCH_CUDA_TAG=${TORCH_CUDA_TAG}
  TORCH_VERSION=${TORCH_VERSION}
  ENV_FILE=${ENV_FILE}
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

Examples:
  # Full setup on a fresh pod
  $(basename "$0") --with-torch --with-sglang --flush-caches

  # Just regenerate env file and verify (no installs)
  $(basename "$0") --doctor-only

  # Pin to CUDA 12.1 instead
  CUDA_VERSION=12.1 TORCH_CUDA_TAG=cu121 $(basename "$0")

After this script finishes, every shell that needs CUDA must:
  source ${ENV_FILE}
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --with-torch)        DO_TORCH=1 ;;
        --with-sglang)       DO_SGLANG=1 ;;
        --flush-caches)      DO_FLUSH_CACHES=1 ;;
        --skip-doctor)       DO_DOCTOR=0 ;;
        --doctor-only)       DO_MINICONDA=0; DO_TOS=0; DO_CUDA_PKGS=0;
                             DO_SYMLINKS=0; DO_ENV_FILE=0; DO_TORCH=0;
                             DO_SGLANG=0; DO_FLUSH_CACHES=0; DO_DOCTOR=1 ;;
        --symlinks-only)     DO_MINICONDA=0; DO_TOS=0; DO_CUDA_PKGS=0;
                             DO_SYMLINKS=1; DO_ENV_FILE=0; DO_TORCH=0;
                             DO_SGLANG=0; DO_FLUSH_CACHES=0; DO_DOCTOR=1 ;;
        --install-gcc13)     INSTALL_GCC13=1 ;;
        --verbose-symlinks)  SYMLINK_VERBOSE=1 ;;
        -h|--help)           usage; exit 0 ;;
        *)                   err "unknown flag: $1"; usage; exit 2 ;;
    esac
    shift
done

#==============================================================================
# ARCH + OS GUARD
#==============================================================================
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64)  MINICONDA_ARCH="x86_64";  CONDA_TARGET_DIR="x86_64-linux" ;;
    aarch64) MINICONDA_ARCH="aarch64"; CONDA_TARGET_DIR="sbsa-linux" ;;
    *) err "Unsupported arch: $ARCH (only x86_64 + aarch64)"; exit 1 ;;
esac

if [ "$(uname -s)" != "Linux" ]; then
    err "This script is Linux-only. Detected: $(uname -s)"
    exit 1
fi
log "Detected: $(uname -s) $ARCH"

#==============================================================================
# 1. MINICONDA
#==============================================================================
install_miniconda() {
    if [ -x "$MINICONDA_PREFIX/bin/conda" ] && \
       "$MINICONDA_PREFIX/bin/conda" --version >/dev/null 2>&1; then
        ok "miniconda already at $MINICONDA_PREFIX"
        return
    fi

    log "Installing miniconda to $MINICONDA_PREFIX (arch=$MINICONDA_ARCH)"

    if [ -d "$MINICONDA_PREFIX" ]; then
        warn "Removing stale/broken $MINICONDA_PREFIX"
        rm -rf "$MINICONDA_PREFIX"
    fi

    local tmp
    tmp="$(mktemp -d)"
    trap "rm -rf '$tmp'" RETURN
    local url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh"
    log "Downloading $url"
    curl -fsSL -o "$tmp/miniconda.sh" "$url"
    bash "$tmp/miniconda.sh" -b -p "$MINICONDA_PREFIX" >/dev/null
    ok "miniconda installed"
}

source_conda() {
    if [ ! -f "$MINICONDA_PREFIX/etc/profile.d/conda.sh" ]; then
        err "Conda profile script missing at $MINICONDA_PREFIX. Run with DO_MINICONDA=1."
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$MINICONDA_PREFIX/etc/profile.d/conda.sh"
    conda activate base
}

#==============================================================================
# 2. ANACONDA TERMS OF SERVICE (required after recent conda updates)
#==============================================================================
accept_tos() {
    log "Accepting Anaconda channel ToS (idempotent)"
    # Older conda doesn't have `tos`; suppress noise on those.
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true
}

#==============================================================================
# 3. CUDA DEV PACKAGES (individual — metapackage trips ClobberError)
#==============================================================================
install_cuda_packages() {
    log "Installing CUDA $CUDA_VERSION dev packages from nvidia channel"

    # Order matters slightly: cudart-dev first so $CONDA_PREFIX/include has
    # the umbrella cuda_runtime.h before others reference it.
    local pkgs=(
        "cuda-cudart-dev=${CUDA_VERSION}"
        "cuda-nvcc=${CUDA_VERSION}"
        "cuda-nvrtc-dev=${CUDA_VERSION}"
        "cuda-cccl=${CUDA_VERSION}"
        "cuda-cuobjdump=${CUDA_VERSION}"
        "cuda-cuxxfilt=${CUDA_VERSION}"
        "cuda-driver-dev=${CUDA_VERSION}"
        # Math libs (flashinfer JIT pulls these in via cutlass utilities)
        "libcurand-dev"
        "libcublas-dev"
        "libcusparse-dev"
        "libcusolver-dev"
    )

    if [ "$INSTALL_GCC13" = "1" ]; then
        log "Also installing gcc=13 + gxx=13 (alternative to -allow-unsupported-compiler)"
        pkgs+=("gcc=13" "gxx=13")
    fi

    conda install -y -c nvidia "${pkgs[@]}" 2>&1 | tail -25 || {
        err "conda install failed — try clearing pkg cache: conda clean -a"
        exit 1
    }
}

#==============================================================================
# 4. HEADER + LIBRARY SYMLINKS
#
# flashinfer's nvcc invocation does:
#   -isystem $CONDA_PREFIX/include
# but the conda nvidia-channel "*-dev" packages (libcurand-dev, libcublas-dev,
# libcusparse-dev, libcusolver-dev, cuda-nvrtc-dev, cuda-cudart-dev, ...) ship
# headers AND .so files under:
#   $CONDA_PREFIX/targets/<TARGET>/include/curand_kernel.h
#   $CONDA_PREFIX/targets/<TARGET>/lib/libcurand.so
# So we link both directories into the canonical $CONDA_PREFIX/{include,lib}.
#
# Build systems (incl. flashinfer JIT linker) expect $CUDA_HOME/lib64; conda
# uses lib/. Symlink lib64 -> lib. Also create unversioned libcudart.so and
# libcuda.so links since linkers do -lcudart / -lcuda.
#
# Idempotent: re-running only creates symlinks that don't already exist.
# Run with --symlinks-only after any new `conda install -c nvidia libxxx-dev`
# to pull its headers/libs into the right place without redoing the full setup.
#==============================================================================
_symlink_into() {
    # _symlink_into <source-file-or-dir> <dest-dir>
    # Skips if dest already exists. Reports verbosely if SYMLINK_VERBOSE=1.
    local src="$1" dest_dir="$2"
    local base
    base="$(basename "$src")"
    local dest="$dest_dir/$base"
    if [ -e "$dest" ] || [ -L "$dest" ]; then
        return 1
    fi
    ln -sf "$src" "$dest"
    [ "$SYMLINK_VERBOSE" = "1" ] && log "  linked $dest -> $src"
    return 0
}

fix_symlinks() {
    local prefix="$MINICONDA_PREFIX"
    local tgt_inc="$prefix/targets/$CONDA_TARGET_DIR/include"
    local tgt_lib="$prefix/targets/$CONDA_TARGET_DIR/lib"

    mkdir -p "$prefix/include" "$prefix/lib"

    # ---- headers --------------------------------------------------------------
    log "Symlinking headers from targets/$CONDA_TARGET_DIR/include/ into include/"
    if [ -d "$tgt_inc" ]; then
        local count=0
        # individual headers (.h, .hpp, .cuh)
        for h in "$tgt_inc"/*.h "$tgt_inc"/*.hpp "$tgt_inc"/*.cuh; do
            [ -f "$h" ] || continue
            _symlink_into "$h" "$prefix/include" && count=$((count + 1))
        done
        # subdirs (cccl, crt, cuda, etc.) — link the dir as a whole
        for d in "$tgt_inc"/*/; do
            [ -d "$d" ] || continue
            _symlink_into "${d%/}" "$prefix/include" && count=$((count + 1))
        done
        ok "Linked $count new header(s)/dir(s) from targets/$CONDA_TARGET_DIR/include/"
    else
        warn "No $tgt_inc — skipping header linking"
    fi

    # ---- libraries from targets/<arch>/lib/ -----------------------------------
    log "Symlinking libs from targets/$CONDA_TARGET_DIR/lib/ into lib/"
    if [ -d "$tgt_lib" ]; then
        local lcount=0
        # All .so* and .a files (covers libcurand.so, libcurand.so.10, etc.)
        for f in "$tgt_lib"/*.so "$tgt_lib"/*.so.* "$tgt_lib"/*.a; do
            [ -e "$f" ] || continue
            _symlink_into "$f" "$prefix/lib" && lcount=$((lcount + 1))
        done
        # CUDA stubs (libcuda.so driver stub usually under targets/<arch>/lib/stubs/)
        if [ -d "$tgt_lib/stubs" ]; then
            for f in "$tgt_lib/stubs"/*.so "$tgt_lib/stubs"/*.so.*; do
                [ -e "$f" ] || continue
                _symlink_into "$f" "$prefix/lib" && lcount=$((lcount + 1))
            done
        fi
        ok "Linked $lcount new lib(s) from targets/$CONDA_TARGET_DIR/lib/"
    else
        warn "No $tgt_lib — skipping lib linking"
    fi

    # ---- lib64 -> lib (linkers default to lib64) ------------------------------
    log "Symlinking lib64 -> lib (if needed)"
    if [ ! -e "$prefix/lib64" ]; then
        ln -sf lib "$prefix/lib64"
        ok "Created $prefix/lib64 -> lib"
    fi

    # ---- unversioned libcudart.so (-lcudart needs the bare name) --------------
    log "Ensuring libcudart.so (unversioned) symlink"
    if [ ! -e "$prefix/lib/libcudart.so" ]; then
        local versioned
        versioned="$(ls -1 "$prefix/lib"/libcudart.so.* 2>/dev/null | head -n1)"
        if [ -n "${versioned:-}" ]; then
            ln -sf "$(basename "$versioned")" "$prefix/lib/libcudart.so"
            ok "Created libcudart.so -> $(basename "$versioned")"
        else
            warn "No libcudart.so.* found — install cuda-cudart-dev"
        fi
    fi

    # ---- libcuda.so driver stub (-lcuda needs it; conda ships only .so.1) -----
    log "Ensuring libcuda.so (driver stub) is reachable"
    if [ ! -e "$prefix/lib/libcuda.so" ]; then
        local stub
        stub="$(find "$prefix" -name 'libcuda.so*' -type f 2>/dev/null | head -n1)"
        if [ -n "${stub:-}" ]; then
            ln -sf "$stub" "$prefix/lib/libcuda.so"
            ok "Linked libcuda.so -> $stub"
        elif [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
            ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$prefix/lib/libcuda.so"
            ok "Linked libcuda.so -> /usr/lib/x86_64-linux-gnu/libcuda.so.1 (system driver)"
        else
            warn "No libcuda.so found anywhere; linker will fail. Install cuda-driver-dev or the host driver."
        fi
    fi
}

#==============================================================================
# 5. SOURCEABLE ENV FILE
#==============================================================================
write_env_file() {
    log "Writing $ENV_FILE"

    cat > "$ENV_FILE" <<EOF
# Generated by scripts/setup_cuda_env.sh on $(date -Iseconds)
# Source this BEFORE running any nvcc / torch / sglang / flashinfer command.
# POSIX-compliant (no bashisms) — works from sh / dash / bash / zsh.

# --- Conda base (needed for CUDA_HOME) ----------------------------------------
export MINICONDA_PREFIX="$MINICONDA_PREFIX"
if [ -f "\$MINICONDA_PREFIX/etc/profile.d/conda.sh" ]; then
    # conda.sh itself uses bash-isms; only \`.\` (POSIX) it under bash/zsh.
    # Under dash this would error, so wrap in a bash check.
    if [ -n "\${BASH_VERSION:-}\${ZSH_VERSION:-}" ]; then
        # shellcheck disable=SC1091
        . "\$MINICONDA_PREFIX/etc/profile.d/conda.sh"
        conda activate base >/dev/null 2>&1 || true
    else
        # Under POSIX shells we can't run conda.sh; set CONDA_PREFIX manually
        # so the rest of this file still works. \`conda\` command won't be on
        # PATH but we don't need it for runtime CUDA env.
        export CONDA_PREFIX="\$MINICONDA_PREFIX"
        export PATH="\$MINICONDA_PREFIX/bin:\$PATH"
    fi
fi

# --- CUDA toolchain -----------------------------------------------------------
export CUDA_HOME="\$CONDA_PREFIX"
export CUDA_PATH="\$CUDA_HOME"
export CUDA_ROOT="\$CUDA_HOME"
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_HOME/lib:\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="\$CUDA_HOME/lib:\$CUDA_HOME/lib64:\${LIBRARY_PATH:-}"
export C_INCLUDE_PATH="\$CUDA_HOME/include:\${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="\$CUDA_HOME/include:\${CPLUS_INCLUDE_PATH:-}"
export CPATH="\$CUDA_HOME/include:\${CPATH:-}"

# --- Compiler flags so JIT (flashinfer / torch-memory-saver) picks up headers
export CFLAGS="-I\$CUDA_HOME/include \${CFLAGS:-}"
export CXXFLAGS="-I\$CUDA_HOME/include \${CXXFLAGS:-}"
export LDFLAGS="-L\$CUDA_HOME/lib -L\$CUDA_HOME/lib64 \${LDFLAGS:-}"
export NVCC_APPEND_FLAGS="\${NVCC_APPEND_FLAGS:-}"

# --- Bypass nvcc gcc-version check (CUDA 12.x rejects gcc>=14 by default) -----
# Safe on Linux distros that ship gcc 14+; gcc 13 and older are unaffected.
export NVCC_PREPEND_FLAGS="-allow-unsupported-compiler \${NVCC_PREPEND_FLAGS:-}"

# --- PyTorch / sglang -------------------------------------------------------
# Architectures we build SGLang/flashinfer JIT kernels for. Adjust if you
# know exactly which GPU you're on (saves JIT compile time).
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"

# Hint flashinfer it can find headers under the conda prefix (some versions
# read CUDA_INCLUDE_DIRS directly).
export CUDA_INCLUDE_DIRS="\$CUDA_HOME/include"
EOF

    ok "Wrote $ENV_FILE"
    log "Source it from now on: source $ENV_FILE"
}

#==============================================================================
# 6. OPTIONAL: TORCH + SGLANG
#==============================================================================
install_torch() {
    log "Reinstalling torch from https://download.pytorch.org/whl/$TORCH_CUDA_TAG"
    if ! command -v uv >/dev/null 2>&1; then
        err "uv not on PATH; install uv first (curl -LsSf https://astral.sh/uv/install.sh | sh)"
        exit 1
    fi
    # Clean up any half-installed torch that lost its __init__.py
    uv pip uninstall -q torch torchvision torchaudio 2>/dev/null || true

    local spec="torch"
    [ -n "$TORCH_VERSION" ] && spec="torch==$TORCH_VERSION"
    uv pip install --index-strategy unsafe-best-match \
        --extra-index-url "https://download.pytorch.org/whl/$TORCH_CUDA_TAG" \
        "$spec"

    log "Verifying torch picked up CUDA"
    uv run python - <<'PY'
import torch
print(f"torch        : {torch.__version__}")
print(f"torch.__file__ {torch.__file__}")
print(f"CUDA build   : {torch.version.cuda}")
print(f"is_available : {torch.cuda.is_available()}")
PY
}

install_sglang() {
    log "Installing sglang[all]>=0.5"
    uv pip install "sglang[all]>=0.5"
}

#==============================================================================
# 7. CACHES
#==============================================================================
flush_caches() {
    log "Flushing JIT caches"
    rm -rf "$HOME/.cache/flashinfer" "$HOME/.cache/tvm" "$HOME/.cache/torch_extensions" 2>/dev/null || true
    ok "Cleared flashinfer / tvm / torch_extensions caches"
}

#==============================================================================
# 8. DOCTOR — verify everything actually works
#==============================================================================
doctor() {
    log "Running verification (doctor)"

    local fails=0
    fail() { err "$1"; fails=$((fails + 1)); }

    # Conda
    if ! command -v conda >/dev/null 2>&1; then
        fail "conda not on PATH"
    fi

    # nvcc
    if ! command -v nvcc >/dev/null 2>&1; then
        fail "nvcc not on PATH (did you 'source $ENV_FILE'?)"
    else
        local v
        v="$(nvcc --version | grep release | sed 's/.*release //; s/,.*//')"
        ok "nvcc release $v"
    fi

    # CUDA_HOME
    if [ -z "${CUDA_HOME:-}" ]; then
        fail "CUDA_HOME unset"
    elif [ ! -d "$CUDA_HOME" ]; then
        fail "CUDA_HOME=$CUDA_HOME does not exist"
    else
        ok "CUDA_HOME=$CUDA_HOME"
    fi

    # Headers — every one we know flashinfer/sglang needs
    local headers=(
        cuda_runtime_api.h
        cuda_runtime.h
        curand.h
        curand_kernel.h
        cublas_v2.h
        cusparse.h
        cusolverDn.h
        nvrtc.h
        cuda.h
    )
    for h in "${headers[@]}"; do
        if [ -e "$CUDA_HOME/include/$h" ]; then
            ok "header: $h"
        else
            # Hint: maybe present under targets/ but not symlinked yet.
            local in_targets
            in_targets="$(find "$CUDA_HOME/targets" -name "$h" 2>/dev/null | head -n1)"
            if [ -n "$in_targets" ]; then
                fail "MISSING header (but present at $in_targets) — run: $(basename "$0") --symlinks-only"
            else
                fail "MISSING header: $CUDA_HOME/include/$h"
            fi
        fi
    done

    # Libs
    local libs=(libcudart.so libcuda.so libcurand.so libcublas.so libcusparse.so libcusolver.so)
    for lib in "${libs[@]}"; do
        if [ -e "$CUDA_HOME/lib/$lib" ] || [ -e "$CUDA_HOME/lib64/$lib" ]; then
            ok "lib: $lib"
        else
            fail "MISSING lib: $lib (looked in lib/ and lib64/)"
        fi
    done

    # lib64 symlink (linkers default to lib64)
    if [ -e "$CUDA_HOME/lib64" ]; then
        ok "lib64 present (-> $(readlink -f "$CUDA_HOME/lib64"))"
    else
        fail "lib64 missing"
    fi

    # Torch — only if installed
    if uv run --no-sync python -c "import torch" 2>/dev/null; then
        local torch_cuda
        torch_cuda="$(uv run --no-sync python -c 'import torch; print(torch.version.cuda or "")')"
        local has_avail
        has_avail="$(uv run --no-sync python -c 'import torch; print(torch.cuda.is_available())')"
        if [ -z "$torch_cuda" ]; then
            warn "torch installed but CPU-only (torch.version.cuda is None) — run with --with-torch"
        else
            ok "torch CUDA: $torch_cuda (is_available=$has_avail)"
        fi
    else
        warn "torch not importable from the active venv (run --with-torch to install)"
    fi

    if [ "$fails" -gt 0 ]; then
        err "Doctor found $fails issue(s) above."
        return 1
    fi
    ok "All checks passed. You're ready for: make run-eagle2 / run-eagle3 / run-sweep"
}

#==============================================================================
# MAIN
#==============================================================================
main() {
    log "spec-eval CUDA setup starting"
    log "CUDA_VERSION=$CUDA_VERSION  MINICONDA_PREFIX=$MINICONDA_PREFIX  TORCH_CUDA_TAG=$TORCH_CUDA_TAG"

    [ "$DO_MINICONDA"     = "1" ] && install_miniconda
    source_conda
    [ "$DO_TOS"           = "1" ] && accept_tos
    [ "$DO_CUDA_PKGS"     = "1" ] && install_cuda_packages
    [ "$DO_SYMLINKS"      = "1" ] && fix_symlinks
    [ "$DO_ENV_FILE"      = "1" ] && write_env_file

    # Bring the freshly-written env into THIS shell so torch/sglang installs
    # and the doctor below all see the right CUDA_HOME.
    if [ -f "$ENV_FILE" ]; then
        # shellcheck disable=SC1090
        source "$ENV_FILE"
    fi

    [ "$DO_TORCH"         = "1" ] && install_torch
    [ "$DO_SGLANG"        = "1" ] && install_sglang
    [ "$DO_FLUSH_CACHES"  = "1" ] && flush_caches
    [ "$DO_DOCTOR"        = "1" ] && doctor

    ok "Done. To use CUDA in a new shell:  source $ENV_FILE"
}

main "$@"
