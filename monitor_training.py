#!/usr/bin/env python3
"""
监控训练进度脚本

实时显示训练日志的最新内容
"""

import time
import os

LOG_FILE = "train_dami_new.log"


def tail_log(n=30):
    """显示日志最后 n 行"""
    if not os.path.exists(LOG_FILE):
        print(f"日志文件不存在: {LOG_FILE}")
        return

    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
        for line in lines[-n:]:
            print(line, end="")


def main():
    print("训练监控 (Ctrl+C 退出)")
    print("=" * 80)

    try:
        while True:
            os.system("clear")
            print("训练监控 - 最新 30 行日志")
            print("=" * 80)
            tail_log(30)
            print("=" * 80)
            print(f"更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(10)  # 每 10 秒刷新一次
    except KeyboardInterrupt:
        print("\n监控已停止")


if __name__ == "__main__":
    main()
