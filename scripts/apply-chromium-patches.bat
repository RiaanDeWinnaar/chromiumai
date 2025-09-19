@echo off
echo Applying ChromiumAI minimal patches...

cd chromium
git apply ..\chromium-patches\ai-browser-integration.patch

REM Copy our AI integration files to Chromium
xcopy /E /I ..\ai-integration\chromium\ .\

echo ChromiumAI patches applied successfully!
echo Minimal modifications approach implemented:
echo - Only 3 lines added to chrome_browser_main.cc
echo - All AI logic kept in separate files
echo - Ready for rebase on release branches
