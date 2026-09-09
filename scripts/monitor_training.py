#!/usr/bin/env python3
"""
监控训练进度脚本

实时显示训练日志的最新内容
"""

import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_FILE = PROJECT_ROOT / "runs" / "logs" / "train_fastsam.log"


def tail_log(log_file, line_count=30):
    """显示日志最后 n 行"""
    if not log_file.is_file():
        print(f"日志文件不存在: {log_file}")
        return

    with log_file.open("r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()
        for line in lines[-line_count:]:
            print(line, end="")


def parse_args():
    """解析日志监控参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_FILE, help="训练日志文件")
    parser.add_argument("--lines", type=int, default=30, help="显示末尾行数")
    parser.add_argument("--interval", type=float, default=10.0, help="刷新间隔（秒）")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.lines <= 0 or args.interval <= 0:
        raise ValueError("显示行数和刷新间隔必须大于 0")
    log_file = args.log.resolve()
    print("训练监控 (Ctrl+C 退出)")
    print("=" * 80)

    try:
        while True:
            print("\033[2J\033[H", end="")
            print(f"训练监控 - 最新 {args.lines} 行日志")
            print("=" * 80)
            tail_log(log_file, args.lines)
            print("=" * 80)
            print(f"更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n监控已停止")


if __name__ == "__main__":
    main()
