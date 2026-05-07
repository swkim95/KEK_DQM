#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# run_all_training.sh
# 데이터 생성 → 학습을 순차적으로 실행 (메모리 부족으로 병렬 불가)
# 실행: bash finetuning/run_all_training.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
SUMMARY_LOG="$LOG_DIR/run_all_${TIMESTAMP}.log"

# 가상환경 활성화 (경로가 다르면 수정)
if [ -f "$PROJECT_DIR/ai/bin/activate" ]; then
    source "$PROJECT_DIR/ai/bin/activate"
fi

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$SUMMARY_LOG"
}

run_step() {
    local step_name="$1"
    local script="$2"
    local step_log="$LOG_DIR/${step_name}_${TIMESTAMP}.log"

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "▶ START : $step_name"
    log "  Script : $script"
    log "  Log    : $step_log"

    local start_ts
    start_ts=$(date +%s)

    if python "$script" 2>&1 | tee "$step_log"; then
        local elapsed=$(( $(date +%s) - start_ts ))
        log "✅ DONE  : $step_name  (elapsed: ${elapsed}s)"
    else
        log "❌ FAILED: $step_name — 전체 파이프라인 중단"
        exit 1
    fi
}

# ──────────────────────────────────────────────────────────────

log "════════════════════════════════════════════════════════════"
log "  AutoTB Full Training Pipeline  ($TIMESTAMP)"
log "════════════════════════════════════════════════════════════"
log "  Project : $PROJECT_DIR"
log "  Python  : $(python --version 2>&1)"

PIPELINE_START=$(date +%s)





# 1. Energy Scan
run_step "EM_data_gen"   "$SCRIPT_DIR/EM_data_gen.py"
run_step "EM_train"      "$SCRIPT_DIR/EM_train.py"

# 2. Calibration
run_step "calib_data_gen" "$SCRIPT_DIR/calib_data_gen.py"
run_step "calib_train"    "$SCRIPT_DIR/calib_train.py"

# 3. HV Equalization
run_step "brain_data_gen" "$SCRIPT_DIR/brain_data_gen.py"
run_step "brain_train"    "$SCRIPT_DIR/brain_train.py"

run_step "hv_data_gen"   "$SCRIPT_DIR/hv_equalization_data_gen.py"
run_step "hv_train"      "$SCRIPT_DIR/hv_equalization_train.py"

# 4. Brain Agent

# ──────────────────────────────────────────────────────────────

TOTAL=$(( $(date +%s) - PIPELINE_START ))
HOURS=$(( TOTAL / 3600 ))
MINS=$(( (TOTAL % 3600) / 60 ))
SECS=$(( TOTAL % 60 ))

log "════════════════════════════════════════════════════════════"
log "  🎉 전체 파이프라인 완료!"
log "  총 소요 시간: ${HOURS}h ${MINS}m ${SECS}s"
log "  요약 로그: $SUMMARY_LOG"
log "════════════════════════════════════════════════════════════"
