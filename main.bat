@echo off
REM 设置标题
title Python Server Manager

REM 使用完整路径运行Python脚本
"E:\code\my_python_server\micromambavenv\python.exe" "E:\code\my_python_server\main.py"

REM 检查运行结果
if %errorlevel% neq 0 (
    echo.
    echo 程序运行出错，错误代码: %errorlevel%
)

echo.
echo 按任意键关闭窗口...
pause >nul