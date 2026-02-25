import os
import datetime
from pathlib import Path
import re

def format_duration(seconds):
    if seconds is None or seconds < 0: return "0:00:00"
    return str(datetime.timedelta(seconds=int(seconds)))

def get_time_info(version_path):
    """
    通过文件名解析开始时间，通过文件属性解析结束时间
    """
    event_files = list(Path(version_path).rglob("events.out.tfevents.*"))
    if not event_files:
        return None, None
    
    start_times = []
    end_times = []

    for ef in event_files:
        # 1. 从文件名提取开始时间 (events.out.tfevents.1767061779.xxxx)
        parts = ef.name.split('.')
        try:
            # 文件名中第4个部分通常是 Unix 时间戳
            if len(parts) > 3:
                start_times.append(float(parts[3]))
        except ValueError:
            pass
        
        # 2. 获取文件的最后修改时间作为可能的结束时间
        end_times.append(ef.stat().st_mtime)

    if not start_times:
        # 如果文件名解析失败，保底使用文件的创建时间
        start_time = min(ef.stat().st_ctime for ef in event_files)
    else:
        start_time = min(start_times)

    # 结束时间取所有 event 文件中最大的那个修改时间
    end_time = max(end_times)
    
    return start_time, end_time

def sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

def analyze_logs(base_dir):
    header = f"{'Experiment':<15} | {'Version':<10} | {'Start Time':<18} | {'Duration'}"
    print(header)
    print("-" * len(header))

    base_path = Path(base_dir)
    # 获取文件夹并排序
    exp_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])

    for experiment_dir in exp_dirs:
        # 获取 version 文件夹并自然排序 (version_1, version_2, version_10...)
        version_dirs = sorted(
            [v for v in experiment_dir.glob("version_*") if v.is_dir()],
            key=lambda x: sort_key(x.name)
        )
        
        for version_dir in version_dirs:
            start, end = get_time_info(version_dir)
            
            if start and end:
                duration_sec = end - start
                # 如果 duration 太小（比如小于1秒），可能是没跑起来
                if duration_sec < 1:
                    duration_str = "Too Short"
                else:
                    duration_str = format_duration(duration_sec)
                
                start_dt = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')
            else:
                duration_str = "No Log File"
                start_dt = "N/A"

            print(f"{experiment_dir.name[:]:<30} | {version_dir.name:<10} | {start_dt:<18} | {duration_str}")

if __name__ == "__main__":
    LOG_ROOT = "/commondocument/group2/ASRCompare/code/train/8B_log"
    if os.path.exists(LOG_ROOT):
        analyze_logs(LOG_ROOT)
    else:
        print(f"Error: Path {LOG_ROOT} does not exist.")