@echo off
REM 推送Lazy Attention到你的GitHub仓库
REM 使用方法：在GitHub创建仓库后，修改下面的REPO_NAME，然后运行此脚本

REM 修改这里：你的GitHub用户名和仓库名
set GITHUB_USER=Fzkuji
set REPO_NAME=adasplash

echo ================================================
echo 推送Lazy Attention到GitHub
echo ================================================
echo.

echo 步骤1：添加新的远程仓库...
git remote add myfork https://github.com/%GITHUB_USER%/%REPO_NAME%.git 2>nul
if errorlevel 1 (
    git remote set-url myfork https://github.com/%GITHUB_USER%/%REPO_NAME%.git
)

echo.
echo 步骤2：检查远程仓库...
git remote -v

echo.
echo 步骤3：推送到你的GitHub...
git push -u myfork main

echo.
echo ================================================
echo ✅ 完成！
echo 访问: https://github.com/%GITHUB_USER%/%REPO_NAME%
echo ================================================

pause
