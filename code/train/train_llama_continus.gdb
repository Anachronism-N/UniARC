# Python 运行脚本的 GDB 调试文件（适用于 train_llama_continus.py）
#
# 使用方法：
# 1) 在项目根目录 /commondocument/group2 下执行：
#    gdb -q -x ASRCompare/code/train/train_llama_continus.gdb
# 2) 程序会在 C 层的 main 处暂停；输入 `run` 开始执行脚本。
# 3) 当命中 Python 执行帧断点时，可用 `next`/`step` 单步；若系统安装了
#    CPython 的 gdb helpers（libpython.py），还可用 `py-bt`/`py-list` 查看
#    Python 层调用栈与当前源码位置。
# 4) 如需为训练脚本传参，编辑下方 `set args` 行，追加所需参数即可。

set pagination off
set confirm off
set breakpoint pending on
set print thread-events off
set disassemble-next-line off

# 让 Python 不缓冲输出，便于观察日志
set env PYTHONUNBUFFERED 1

# 为脚本提供项目路径，避免 import 问题（根据需要调整）
# 若你从 /commondocument/group2 目录启动 gdb，此相对路径即可生效。
set env PYTHONPATH ./ASRCompare

# 选择要调试的可执行程序：使用 PATH 中的 python3（conda 环境会优先）
file python3

# 训练脚本及其参数（按需追加）
set args ASRCompare/code/train/train_llama_continus.py

# 尝试加载 CPython 的 gdb helper（若不存在会提示，可忽略）
source /usr/share/gdb/python/libpython.py

# 在关键 Python/C 函数处设置断点，便于进入 Python 执行循环
break PyRun_SimpleFileExFlags
break PyEval_EvalFrameDefault
break _PyEval_EvalFrameDefault

# 在异常设置处断下，便于定位错误来源
break PyErr_SetObject
break _PyErr_SetObject

# 启动到 main 位置，便于进一步 run/step
start

# 每次停下时给出提示（不依赖额外插件）
define hook-stop
  echo \n[STOP] 使用 `next`/`step` 进行单步；若已加载 libpython helpers，可用 `py-bt`/`py-list` 查看 Python 层。\n
end