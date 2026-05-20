#!/bin/bash

cd /opt/deepCheckins || exit 1

echo "[deepCheckins] starting services..."

# start all processes in background
/usr/bin/python3 main.py &
PID1=$!

/usr/bin/python3 motion.py &
PID2=$!

/usr/bin/python3 app.py &
PID3=$!

echo "PIDs: $PID1 $PID2 $PID3"

# wait so systemd keeps service alive
wait $PID1 $PID2 $PID3