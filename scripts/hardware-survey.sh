#!/bin/sh
set -u

REPORT="${1:-bmass-hardware-survey.txt}"

{
  echo "=========================================="
  echo " BMASS HARDWARE SURVEY"
  echo "=========================================="
  echo
  echo "Date:"
  date
  echo
  echo "========== OPERATING SYSTEM =========="
  cat /etc/os-release 2>/dev/null || true
  echo
  echo "========== KERNEL =========="
  uname -a
  echo
  echo "========== HOSTNAME =========="
  hostname
  echo
  echo "========== CPU =========="
  if command -v lscpu >/dev/null 2>&1; then lscpu; else cat /proc/cpuinfo; fi
  echo
  echo "========== MEMORY =========="
  if command -v free >/dev/null 2>&1; then free -h; else cat /proc/meminfo; fi
  echo
  echo "========== STORAGE =========="
  df -h
  echo
  echo "========== BLOCK DEVICES =========="
  command -v lsblk >/dev/null 2>&1 && lsblk || true
  echo
  echo "========== GPU / PCI =========="
  if command -v lspci >/dev/null 2>&1; then
    lspci | grep -Ei 'vga|3d|display|nvidia|amd|intel' || echo "No display adapter matched"
  else
    echo "lspci not installed"
  fi
  echo
  echo "========== PYTHON =========="
  python3 --version 2>&1 || echo "Python 3 not installed"
  echo
  echo "========== NUMPY =========="
  python3 -c 'import numpy; print(numpy.__version__)' 2>&1 || echo "NumPy not installed"
  echo
  echo "========== COMPILER =========="
  gcc --version 2>/dev/null | head -n 1 || echo "GCC not installed"
  echo
  echo "=========================================="
  echo " End of Report"
  echo "=========================================="
} | tee "$REPORT"
