#!/bin/zsh
# Cron wrapper for the public-dump poller. Runs the poll cycle, logs with a timestamp.
# Auto-expires after the 2026-07-15 competition deadline (no-op past it).
# Register with:  crontab -e  ->  0 */3 * * * /Users/minseong/project/neurogolf/tools/poll_dumps_cron.sh
REPO="/Users/minseong/project/neurogolf"
LOG="$REPO/state/poll_dumps.log"
export PATH="/Users/minseong/.local/bin:/opt/homebrew/Caskroom/miniconda/base/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# deadline guard (YYYYMMDD)
if [ "$(date +%Y%m%d)" -gt 20260715 ]; then
  exit 0
fi

cd "$REPO" || exit 1
echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG"
uv run python tools/poll_public_dumps.py >> "$LOG" 2>&1
echo "" >> "$LOG"
