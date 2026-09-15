# Koakumix 一键安装（npm 式）：editable 安装 + 全局命令注册 + 自检
#
# 用法（任意目录）：
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   powershell -ExecutionPolicy Bypass -File install.ps1 -RepoRoot G:\C\PYT\qlh -Python <python.exe>
#   powershell -ExecutionPolicy Bypass -File install.ps1 -NoPath     # 只写 shim，不改 PATH
#
# 与 Patchouli 的 install.ps1 同构（%USERPROFILE%\bin + User PATH，幂等）。
# 区别：shim 里显式 pushd 到仓库根并调用 python -m <module>，所以不依赖 editable 安装的
# 路径注入是否成功（本仓的 package-dir 布局曾让 editable finder 生成空映射）。
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Python = "",
    [switch]$NoPath
)

$ErrorActionPreference = "Stop"
if (-not $Python) {
    $venvPy = Join-Path $RepoRoot ".venv-test\Scripts\python.exe"
    if (Test-Path $venvPy) { $Python = $venvPy } else { $Python = (Get-Command python).Source }
}
Write-Host "== Koakumix 安装 ==" -ForegroundColor Red
Write-Host "RepoRoot : $RepoRoot"
Write-Host "Python   : $Python"

# 1) editable 安装（提供 koakumix / koakumix-tui / koakumix-desktop console scripts）
& $Python -m pip install -e (Join-Path $RepoRoot "harness_workbench") --quiet
Write-Host "[1/3] editable 安装完成（harness_workbench）"

# 2) 全局命令 shim：%USERPROFILE%\bin\*.cmd
$binDir = Join-Path $env:USERPROFILE "bin"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null

function Write-Shim([string]$Name, [string]$Module) {
    $shim = Join-Path $binDir "$Name.cmd"
    # 纯 ASCII、无中文，避免 Set-Content -Encoding ASCII 下出现乱码。
    $content = "@echo off`r`npushd `"$RepoRoot`"`r`n`"$Python`" -m $Module %*`r`nset `"KOAKUMIX_EXIT=%ERRORLEVEL%`"`r`npopd`r`nexit /b %KOAKUMIX_EXIT%`r`n"
    Set-Content -Path $shim -Value $content -Encoding ASCII
    Write-Host "  shim: $shim  ->  python -m $Module"
}

Write-Shim "koakumix" "harness_workbench.cli"
Write-Shim "koakumix-tui" "harness_workbench.tui"
Write-Shim "koakumix-desktop" "harness_workbench.desktop"
Write-Host "[2/3] 全局命令已注册（koakumix / koakumix-tui / koakumix-desktop）"

# 3) PATH 注册：把 bin 放到 User PATH 的**最前**（幂等）。
#    只"追加到末尾"是不够的 —— 系统 Python / conda 的 Scripts 目录常排在前面，若其中恰有
#    同名 exe（例如早前误装到系统 Python 的 koakumix.exe），它会抢先命中、且可能已损坏。
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($NoPath) {
    Write-Host "[3/3] 跳过 PATH 注册（-NoPath）"
} else {
    $parts = @($userPath -split ';' | Where-Object { $_ -and ($_ -ne $binDir) })
    $newPath = (@($binDir) + $parts) -join ';'
    if ($newPath -ne $userPath) {
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
        Write-Host "[3/3] PATH 已置顶: $binDir（新终端生效）"
    } else {
        Write-Host "[3/3] PATH 已在最前: $binDir"
    }
}

# 自检：直接调用刚写的 shim（不依赖 PATH 是否已在新进程中生效）
Write-Host ""
Write-Host "自检（koakumix --help）："
& (Join-Path $binDir "koakumix.cmd") --help | Select-Object -First 6
Write-Host ""
Write-Host "完成。命令：" -ForegroundColor Red
Write-Host "  koakumix-tui        恐虐配色 CLI 工作台（默认连 http://127.0.0.1:8090 的 harness API；--no-splash 可关启动动画）"
Write-Host "  koakumix            harness API / CLI（--serve-port 默认 8090）"
Write-Host "  koakumix-desktop    pywebview 桌面壳（默认 backend=qlh，连 8000 主项目）"
