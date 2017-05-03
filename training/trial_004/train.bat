@echo off

for /l %%i in (1, 1, 5) do (
    python stock_classification.py %1 %%i
)