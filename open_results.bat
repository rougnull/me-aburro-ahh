@echo off
REM Script para abrir visualizaciones de la simulación NeuroMechFly

setlocal enabledelayedexpansion

echo ========================================
echo NeuroMechFly Visualization Viewer
echo ========================================
echo.
echo Archivos generados en: data\20260214_225011\
echo.
echo 1. 3D Trajectory (RECOMENDADO - Ver primero)
echo    %UserProfile%\Documents\Workspace\NeuroMechFly Sim\data\20260214_225011\3d_trajectory.png
echo.
echo 2. Behavior Analysis (Análisis 4-panel)
echo    %UserProfile%\Documents\Workspace\NeuroMechFly Sim\data\20260214_225011\behavior_analysis.png
echo.
echo 3. Neural Heatmap (Actividad por capa)
echo    %UserProfile%\Documents\Workspace\NeuroMechFly Sim\data\20260214_225011\neural_heatmap.png
echo.
echo 4. Trajectory 2D (Vista superior)
echo    %UserProfile%\Documents\Workspace\NeuroMechFly Sim\data\20260214_225011\trajectory.png
echo.
echo 5. Neural Activity (Raster de DNs)
echo    %UserProfile%\Documents\Workspace\NeuroMechFly Sim\data\20260214_225011\neural_activity.png
echo.
echo 6. Odor Response (Input olfativo)
echo    %UserProfile%\Documents\Workspace\NeuroMechFly Sim\data\20260214_225011\odor_response.png
echo.
echo 7. Data (HDF5 - Cargar en Python)
echo    simulation_data.h5 (11.2 MB)
echo.
echo ========================================
echo Abriendo archivos principales en visor...
echo ========================================
echo.

cd /d "C:\Users\eduar\Documents\Workspace\NeuroMechFly Sim\data\20260214_225011"

REM Abrir principales
start 3d_trajectory.png
start behavior_analysis.png
start neural_heatmap.png

echo.
echo Archivos abiertos en el visor de imágenes.
echo.
pause
