#!/bin/bash
# Launcher: ask for a ticker, update config.py, train the model, open the dashboard
#Run from the root project: ./launcher.sh

read -p "Enter ticker symbol: " TICKER
TICKER=$(echo "$TICKER" | tr '[:lower:]' '[:upper:]' | xargs) # uppercase + trim

if [ -z "$TICKER" ]; then
    echo "No ticker entered - exiting"
    exit 1
fi

echo ">>>Updating config.py: TICKER = \"$TICKER\""
if grep -qE '^TICKER[[:space:]]*=' config,.py; then
    # Replace the existing ticker line
    sed -i "s/^TICKER[[:space:]]*=.*/TICKER = \"$TICKER\"/" config.py
else
    # No TICKER line yet - append one
    echo "TICKER = \"$TICKER\"" >> config.py
fi

echo ">>> Training model for $TICKER..."
python -m src.rl_train --ticker "$TICKER" || { echo "Training failed — not launching dashboard."; exit 1; }
 
echo ">>> Launching dashboard..."
streamlit run app/dashboard.py
 
