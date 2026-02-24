#!/bin/bash

# Function to handle script termination
cleanup() {
    echo "Stopping all processes..."
    # Kill all child processes of this script
    pkill -P $$
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# Start Python server
echo "Starting Python Server..."
python3 server.py &

# Wait a moment for the server to initialize
sleep 2

# Start React app
echo "Starting React Application..."
cd lebag && npm run dev &

# Wait for all background processes to finish
wait
