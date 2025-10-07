@echo off
echo 正在启动三个服务...

:: 设置相同的任务组ID，这样可以统一管理
set TASK_GROUP=MyServerGroup

:: 打开第一个终端运行 see_server
start "Python SSE Server" /d "E:\code\my_python_server" cmd /k "title Python SSE Server & micromambavenv\python sse_server.py & taskkill /F /FI \"TASKGROUP eq %TASK_GROUP%\""

:: 打开第二个终端运行 excel_mcp
start "Excel MCP Server" /d "E:\code\excel-mcp-server" cmd /k "title Excel MCP Server & uvx excel-mcp-server sse & taskkill /F /FI \"TASKGROUP eq %TASK_GROUP%\""

:: 打开第三个终端运行 llm_server
start "LLM Server" /d "E:\code\my_python_server" cmd /k "title LLM Server & micromambavenv\python llm_server.py & taskkill /F /FI \"TASKGROUP eq %TASK_GROUP%\""

echo 已打开三个终端窗口：
echo 1. Python SSE Server 终端
echo 2. Excel MCP Server 终端
echo 3. LLM Server 终端
echo.
echo 注意：关闭任一终端窗口时，其他所有终端也会被关闭
echo 按任意键继续...
pause