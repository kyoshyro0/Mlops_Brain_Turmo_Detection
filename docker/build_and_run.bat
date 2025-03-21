@echo off
cd ..
docker-compose -f docker/docker-compose.yml build --no-cache
docker-compose -f docker/docker-compose.yml up -d
echo Container đang chạy ở http://localhost:8501 (Frontend) và http://localhost:8000 (API)