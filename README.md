# Deep Reinforcement Learning for Leverage Trading

An exploratory learning project implementing a deep reinforcement learning approach for leveraged trading, combining sentiment analysis with technical indicators using Alpaca for market data access.

## Quick Start

1. **Prerequisites**: Python 3.8+, Jupyter, free [Alpaca account](https://alpaca.markets)
2. **Install UV**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. **Setup environment**:
   ```bash
   uv python install 3.11
   uv venv && source .venv/bin/activate
   uv pip install ray[rllib] gymnasium pandas numpy transformers alpaca-py talib accelerate bitsandbytes peft sentencepiece jupyter
   ```
4. **Configure Alpaca API**: Create `Alpaca API Key.txt` and `Alpaca API Secret.txt` files
5. **Run notebooks in order**: `fetch_market_data.ipynb` → `generate_sentiment_scores.ipynb` → `train_and_inference.ipynb` → `evaluate_performance.ipynb`

## Project Structure

- `train_and_inference.ipynb`: Main RL training and inference pipeline
- `evaluate_performance.ipynb`: Performance benchmarking and visualization
- `fetch_market_data.ipynb`: Alpaca API data acquisition with technical indicators
- `generate_sentiment_scores.ipynb`: Transformer-based news sentiment analysis
- `Project Report.pdf`: Comprehensive technical documentation

## Features

**Core Components**:
- Ray RLlib-powered reinforcement learning for trading decisions
- Transformer models for financial news sentiment analysis
- Technical indicators (EMA, RSI, MFI, Directional Movement)
- Built-in leverage mechanism for position sizing

**Analysis & Evaluation**:
- Backtesting against buy-and-hold and market benchmarks
- Risk metrics: Sharpe ratio, maximum drawdown, win rate
- Performance visualization and comparative analysis

## Data & Configuration

**Data Generation** (no large files included):
- Market data: Run `fetch_market_data.ipynb` with Alpaca credentials
- News sentiment: Run `generate_sentiment_scores.ipynb` on financial news
- Technical indicators: Automatically calculated during data processing

**API Setup**:
1. Register at [alpaca.markets](https://alpaca.markets)
2. Get API keys from dashboard
3. Create credential files:
   - `Alpaca API Key.txt`: Your API key
   - `Alpaca API Secret.txt`: Your API secret

## System Requirements

- **RAM**: 8GB+ recommended (16GB+ for GPU acceleration)
- **Python**: 3.8+ (3.11 recommended)
- **GPU**: CUDA-compatible for sentiment analysis acceleration
- **Storage**: ~500MB for dependencies + data generation

## Usage Workflow

1. **Data Acquisition**: Fetch historical market data and compute technical indicators
2. **Sentiment Processing**: Generate sentiment scores from financial news
3. **Model Training**: Train RL agent with combined market and sentiment features
4. **Performance Evaluation**: Compare against benchmarks and analyze risk metrics

## Troubleshooting

**Common Issues**:
- **API errors**: Verify Alpaca credentials and account permissions
- **Memory issues**: Reduce batch sizes or use CPU for sentiment analysis
- **UV not found**: Restart terminal after installation
- **Jupyter kernels**: Run `uv pip install ipykernel` and restart Jupyter

**Performance Tips**:
- Use GPU for faster sentiment analysis training
- Start with smaller datasets for initial testing
- Monitor Alpaca API rate limits during data fetching

## Disclaimer

This project is for educational and research purposes only. Not financial advice. Trading involves substantial risk. Past performance ≠ future results. Consult professionals before investing.

