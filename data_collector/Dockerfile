FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for QuantLib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY data_collector/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code into the image
COPY btc_pricer/ btc_pricer/
COPY scripts/ scripts/
COPY data_collector/ data_collector/
COPY cli_terminal.py cli.py config.yaml ./

# Create results directory
RUN mkdir -p data_collector/results

CMD ["python", "-m", "data_collector.collector"]
