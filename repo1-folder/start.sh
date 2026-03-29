#!/bin/bash
echo "============================================"
echo "  FaceGuard v2 - Face Mask Detection System"
echo "============================================"

echo "[1/3] Installing Python dependencies..."
cd backend && pip install -r requirements.txt && cd ..

echo ""
echo "[2/3] Installing Node dependencies..."
cd frontend && npm install && cd ..

echo ""
echo "[3/3] Starting servers..."

cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACK=$!
sleep 3

cd ../frontend && npm start &
FRONT=$!

echo ""
echo "============================================"
echo "  Open: http://localhost:3000"
echo "  NOTE: First run downloads AI models (~300MB)"
echo "============================================"

trap "kill $BACK $FRONT" SIGINT
wait
