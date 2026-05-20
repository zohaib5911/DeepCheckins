#!/usr/bin/env python3

import os
import shutil
import argparse

import paths

DEFAULT_UNKNOWN_DIR       = paths.UNKNOWN_DIR
DEFAULT_CHECKINS          = paths.KNOWN_LOG_FILE
DEFAULT_DEBUG_LOG         = paths.DEBUG_LOG_FILE
DEFAULT_NOFACE_DIR        = paths.NOFACE_DIR
DEFAULT_DEBUG_FAIL_DIR    = paths.DEBUG_FAIL_DIR
DEFAULT_KNOWN_DETECT_DIR  = paths.KNOWN_DETECT_DIR
DEFAULT_UNKNOWN_DEBUG_DIR = paths.UNKNOWN_DEBUG_DIR


def remove_unknown_dir(path: str = DEFAULT_UNKNOWN_DIR) -> None:
	if os.path.exists(path):
		shutil.rmtree(path)
		print(f"[RESET] Removed unknown folder: {path}")
	else:
		print(f"[RESET] Unknown folder not found: {path}")


def remove_checkins_file(path: str = DEFAULT_CHECKINS) -> None:
	if os.path.exists(path):
		os.remove(path)
		print(f"[RESET] Removed checkins file: {path}")
	else:
		print(f"[RESET] Checkins file not found: {path}")


def remove_debug_log(path: str = DEFAULT_DEBUG_LOG) -> None:
	if os.path.exists(path):
		os.remove(path)
		print(f"[RESET] Removed debug log: {path}")
	else:
		print(f"[RESET] Debug log not found: {path}")


def remove_dir(path: str, label: str) -> None:
	if os.path.exists(path):
		shutil.rmtree(path)
		print(f"[RESET] Removed {label}: {path}")
	else:
		print(f"[RESET] {label} not found: {path}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Reset utility for DeeepScanIN")
	parser.add_argument("--unknown-dir", default=DEFAULT_UNKNOWN_DIR, help="Path to unknown folder")
	parser.add_argument("--checkins", default=DEFAULT_CHECKINS, help="Path to checkins.txt")
	parser.add_argument("--debug-log", default=DEFAULT_DEBUG_LOG, help="Path to debug log")
	parser.add_argument("--noface-dir", default=DEFAULT_NOFACE_DIR, help="Path to noface folder")
	parser.add_argument("--debug-fail-dir", default=DEFAULT_DEBUG_FAIL_DIR, help="Path to debug failures folder")
	parser.add_argument("--known-detect-dir", default=DEFAULT_KNOWN_DETECT_DIR, help="Path to known detections folder")
	parser.add_argument("--unknown-debug-dir", default=DEFAULT_UNKNOWN_DEBUG_DIR, help="Path to unknown debug folder")
	parser.add_argument("--all", action="store_true", help="Remove unknown folder, checkins.txt, and debug log")
	parser.add_argument("--unknown", action="store_true", help="Remove unknown folder only")
	parser.add_argument("--checkins-only", action="store_true", help="Remove checkins.txt only")

	args = parser.parse_args()

	if args.all or (not args.unknown and not args.checkins_only):
		remove_unknown_dir(args.unknown_dir)
		remove_checkins_file(args.checkins)
		remove_debug_log(args.debug_log)
		remove_dir(args.noface_dir, "noface folder")
		remove_dir(args.debug_fail_dir, "debug failures folder")
		remove_dir(args.known_detect_dir, "known detections folder")
		remove_dir(args.unknown_debug_dir, "unknown debug folder")
		return

	if args.unknown:
		remove_unknown_dir(args.unknown_dir)

	if args.checkins_only:
		remove_checkins_file(args.checkins)

	if not args.unknown and not args.checkins_only:
		remove_debug_log(args.debug_log)
		remove_dir(args.noface_dir, "noface folder")
		remove_dir(args.debug_fail_dir, "debug failures folder")
		remove_dir(args.known_detect_dir, "known detections folder")
		remove_dir(args.unknown_debug_dir, "unknown debug folder")


if __name__ == "__main__":
	main()
