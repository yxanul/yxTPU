#!/usr/bin/env bash
# Fleet control for the multi-host TPU slice: gcloud ssh as a CONTROL PLANE,
# never as a data plane.
#
# Mental model (2026-08-17, after a day of dead shells):
#   * one ssh per worker, all workers IN PARALLEL, each with a hard wall-clock
#     limit and TCP keepalives - a wedged host fails in seconds instead of
#     hanging the whole `--worker=all` fan-out forever;
#   * long-running work is LAUNCHED detached (setsid nohup, per-worker log,
#     stdin from /dev/null) and the ssh returns immediately;
#   * progress is OBSERVED by reading log files in short-lived sessions;
#     silence means "unknown", never "dead" - check procs + log mtime;
#   * the same script that launches also knows how to find orphans
#     (spawned producers, stale shells) and to report disk/health, because
#     "idle" must be verified, not assumed.
#
# Usage:
#   scripts/fleet.sh status                      # per-worker: load, disk, procs, log tail
#   scripts/fleet.sh run  "<cmd>"                # short command on every worker (60 s cap)
#   scripts/fleet.sh launch <name> "<cmd>"       # detached; log ~/fleet-<name>.log
#   scripts/fleet.sh tail <name> [N]             # last N lines of that log per worker
#   scripts/fleet.sh procs                       # trainer / producer / probe processes
#   scripts/fleet.sh kill-orphans                # spawned producers whose trainer is gone
#   scripts/fleet.sh health                      # healthagent RSS + kernel OOM-log growth
#   FLEET_WORKERS="0 3" scripts/fleet.sh status  # subset
set -uo pipefail

TPU_NAME="${TPU_NAME:-yxtpu-v4-64-train}"
TPU_ZONE="${TPU_ZONE:-us-central2-b}"
WORKERS="${FLEET_WORKERS:-0 1 2 3 4 5 6 7}"
SSH_TIMEOUT="${FLEET_SSH_TIMEOUT:-60}"          # wall clock per worker, seconds
SSH_OPTS=(-o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o BatchMode=yes)
# Any yxtpu process that owns producers/chips: train, profile, benchmark,
# calibrate-mix, sft, the eval scripts. Used by status/launch/kill-orphans
# so that "trainer running" means the same thing everywhere (the bracket
# trick keeps grep from matching itself).
TRAINER_PATTERN='[y]x-pretrain|[y]xtpu_pretrain'

_one() {  # _one <worker> <cmd> - hard wall-clock cap without GNU timeout (macOS)
  local w="$1" cmd="$2" out rc
  out=$( gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --worker="$w" \
      --command="$cmd" -- "${SSH_OPTS[@]}" 2>&1 & pid=$!
    ( sleep "$SSH_TIMEOUT"; pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null; sleep 2; pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null ) 2>/dev/null & wd=$!
    wait "$pid" 2>/dev/null; rc=$?
    kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
    if [ $rc -ge 128 ]; then
      if [ "${FLEET_EXPECT_HANG:-0}" = 1 ]; then echo "(session cut after ${SSH_TIMEOUT}s - expected for a detached launch; verify with 'procs')"
      else echo "!! worker $w: TIMEOUT after ${SSH_TIMEOUT}s (host wedged or unreachable)"; fi
    fi
    exit 0 ) 
  printf '%s\n' "$out" | grep -v '^SSH:\|Using ssh\|^$' | sed "s/^/[w$w] /"
}

_all() {  # _all <cmd>  - parallel over workers, output grouped per worker
  local cmd="$1"; local pids=() tmp
  tmp=$(mktemp -d)
  for w in $WORKERS; do ( _one "$w" "$cmd" > "$tmp/$w" ) & pids+=($!); done
  wait "${pids[@]}" 2>/dev/null
  for w in $WORKERS; do cat "$tmp/$w"; done
  rm -rf "$tmp"
}

case "${1:-}" in
  status)
    _all 'echo "$(hostname) up=$(uptime | sed "s/.*load average/load/") disk=$(df -h / | awk "NR==2{print \$4\" free\"}") trainers=$(ps -eo cmd | grep -Ec "'"$TRAINER_PATTERN"'") producers=$(pgrep -fc "spawn_mai[n]") oomlog=$(sudo grep -c "Out of memory" /var/log/kern.log 2>/dev/null)"; for f in ~/fleet-*.log ~/smoke_*.log; do [ -f "$f" ] && echo "  $(basename $f) $(date -r "$f" +%H:%M:%S) $(tail -c 300 "$f" | tail -1 | cut -c1-120)"; done; true' ;;
  run)
    _all "$2" ;;
  launch)
    # gcloud's ssh keeps the channel open while the detached child lives, so
    # the session is cut by the cap once "launched" has been printed.
    # Refuses when a yxtpu process is already running on the worker: the
    # libtpu lockfile is removed for the launch (a crashed run leaves it
    # behind), so it would not protect a live run from a second one.
    # The command is passed base64-encoded so quotes survive the ssh hop.
    name="$2"; cmd="$3"; export FLEET_EXPECT_HANG=1; SSH_TIMEOUT="${FLEET_LAUNCH_TIMEOUT:-25}"
    enc=$(printf '%s' "$cmd" | base64 | tr -d '\n')
    _all "if [ \"\$(ps -eo cmd | grep -Ec '$TRAINER_PATTERN')\" != 0 ]; then echo '!! refused: a yxtpu process is running (see procs); stop it first'; exit 0; fi; cd ~/yxTPU/pretraining && export PATH=\$HOME/.local/bin:\$PATH && rm -f /tmp/libtpu_lockfile && echo '$enc' | base64 -d > ~/fleet-$name.cmd && PYTHONUNBUFFERED=1 setsid nohup bash ~/fleet-$name.cmd > ~/fleet-$name.log 2>&1 < /dev/null & disown; sleep 1; echo launched pid=\$! log=~/fleet-$name.log" ;;
  tail)
    name="$2"; n="${3:-3}"
    _all "tail -n $n ~/fleet-$name.log 2>/dev/null | cut -c1-240" ;;
  procs)
    _all 'ps -eo pid,etime,pcpu,rss,cmd | grep -E "[y]x-pretrain|[s]pawn_main|[h]2d_probe|[y]xtpu_pretrain" | cut -c1-140' ;;
  kill-orphans)
    # spawned producers left behind when no yxtpu process (trainer, profile,
    # benchmark, calibrate-mix, sft, eval) is running on the worker at all
    _all 'if [ "$(ps -eo cmd | grep -Ec "'"$TRAINER_PATTERN"'")" = 0 ]; then n=$(pgrep -fc "spawn_mai[n]|resource_tracke[r]"); pgrep -f "spawn_mai[n]|resource_tracke[r]" | xargs -r kill -TERM; sleep 2; pgrep -f "spawn_mai[n]|resource_tracke[r]" | xargs -r kill -KILL; echo "killed $n orphans"; else echo "yxtpu process running - not touching producers"; fi' ;;
  health)
    _all 'echo "healthagent=$(systemctl is-active healthagent.service) rss=$(ps -eo rss,cmd | grep "[h]ealthAgent" | grep -v docker | awk "{print \$1}")KB oomlog=$(sudo grep -c "Out of memory" /var/log/kern.log 2>/dev/null) kern.log=$(sudo du -h /var/log/kern.log 2>/dev/null | cut -f1) syslog=$(sudo du -h /var/log/syslog 2>/dev/null | cut -f1) disk=$(df -h / | awk "NR==2{print \$5}")"' ;;
  *)
    sed -n '2,25p' "$0"; exit 1 ;;
esac
