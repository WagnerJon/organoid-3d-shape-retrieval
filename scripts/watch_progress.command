#!/bin/zsh
cd "${0:A:h}/.."
echo "Live cpsam_v2 reconstruction progress — Ctrl+C closes this viewer; processing continues."
echo "Summary: results/cpsam_v2_remaining/status.json"
echo ""
tail -n 30 -F results/cpsam_v2_remaining/progress.log
