# MICRO-ASXR Runtime Server

## MX2LM MICRONAUT AI - Single-Page Runtime with REST API & NPX Server

A comprehensive single-page runtime with REST API that can train with Google Colab and serve as a CPU/GPU backend server/API host. Built on the ASX language framework.

```
🚀 MICRO-ASXR
├── Single-page runtime interface
├── Comprehensive REST API
├── NPX CLI server
├── N-gram language models
├── Transformer model support
├── Google Colab training integration
├── CPU/GPU backend support
└── WebSocket real-time updates
```

## Features

- **Single-Page Runtime**: Modern web interface for model interaction
- **REST API**: Full-featured API for predictions, training, and model management
- **NPX Server**: Run instantly with `npx @asxr/runtime-server`
- **Dual Model Support**: Both n-gram and transformer models
- **Colab Training**: Train models on Google Colab with GPU
- **CPU/GPU Runtime**: Flexible backend switching
- **WebSocket Support**: Real-time updates for training progress
- **Model Management**: Load, unload, and manage multiple models

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/cannaseedus-bot/MICRO-ASXR.git
cd MICRO-ASXR

# Install dependencies
npm install

# Start the server
npm start
```

### Using NPX (No Installation)

```bash
# Run directly with NPX
npx @asxr/runtime-server start

# Or use the CLI
npx @asxr/runtime-server --help
```

## Usage

### Start Server

```bash
# Production mode
npm start

# Development mode (auto-reload)
npm run dev

# Using CLI
asxr start
```

The server will start on `http://localhost:3000`

### Web Interface

Open your browser to `http://localhost:3000` to access the runtime interface:

- **Predict Tab**: Test predictions with the n-gram model
- **Train Tab**: Train the model with new sentences
- **Models Tab**: View and manage loaded models
- **Config Tab**: Configure runtime settings (CPU/GPU)
- **API Tab**: API documentation

### CLI Commands

```bash
asxr start              # Start the server
asxr dev                # Start in dev mode
asxr train <dataset>    # Train a model
asxr predict <text>     # Make a prediction
asxr init <project>     # Initialize new project
asxr version            # Show version
asxr help               # Show help
```

## REST API

### Health & System

```bash
# Health check
GET /api/health

# System information
GET /api/system

# Get ASX configuration
GET /api/config
```

### Prediction

```bash
# N-gram prediction
POST /api/predict
{
  "text": "hello world",
  "maxResults": 5
}

# Python inference
POST /api/infer
{
  "text": "hello world",
  "runtime": "gpu"
}
```

### Training

```bash
# Train n-gram model
POST /api/train/ngram
{
  "sentences": [
    "machine learning is amazing",
    "artificial intelligence is powerful"
  ]
}

# Start training job
POST /api/train/start
{
  "dataset": ["sentence 1", "sentence 2"],
  "modelType": "ngram",
  "runtime": "gpu",
  "epochs": 10
}

# Get training status
GET /api/train/status/:jobId
```

### Models

```bash
# List models
GET /api/models

# Load model
POST /api/models/load
{
  "modelId": "my-model",
  "type": "ngram",
  "path": "brain/bigrams.json"
}

# Unload model
DELETE /api/models/:modelId
```

### Runtime Control

```bash
# Get runtime info
GET /api/runtime

# Set runtime (CPU/GPU)
POST /api/runtime
{
  "runtime": "gpu"
}
```

### Brain Data

```bash
# Get bigrams
GET /api/brain/bigrams

# Get trigrams
GET /api/brain/trigrams

# Update brain data
POST /api/brain/bigrams
{ ... bigram data ... }
```

## Training with Google Colab

### 1. Upload Notebook

Upload `colab_training.ipynb` to Google Colab

### 2. Run Training

```python
# In Colab, run all cells to:
# 1. Install dependencies
# 2. Prepare dataset
# 3. Train n-gram model (CPU)
# 4. Train transformer model (GPU)
# 5. Export models
```

### 3. Download Models

```python
# Download trained models
files.download('brain.zip')
files.download('asx-gpt2.zip')
```

### 4. Deploy to Server

```bash
# Extract models
unzip brain.zip -d ./
unzip asx-gpt2.zip -d ./

# Restart server
npm start
```

## Python Scripts

### Training

```bash
# Train n-gram model
python3 python/asx_train.py dataset.txt --type ngram

# Train transformer model
python3 python/asx_train.py dataset.txt \
  --type transformer \
  --epochs 5 \
  --batch-size 4 \
  --gpu
```

### Inference

```bash
# N-gram prediction
python3 python/asx_infer.py "hello world" --type ngram

# Transformer generation
python3 python/asx_infer.py "hello world" \
  --type transformer \
  --max-length 50 \
  --gpu
```

### Build N-grams

```bash
# Build from corpus
python3 python/asx_ngram_builder.py build corpus.txt

# Merge multiple models
python3 python/asx_ngram_builder.py merge brain1/ brain2/ --output brain/
```

## Configuration

### Environment Variables

```bash
PORT=3000                    # Server port
HOST=0.0.0.0                 # Server host
CUDA_VISIBLE_DEVICES=0       # GPU device
```

### ASX Configuration

Edit `MICRO.ASXR` to customize:

- System metadata
- Agent configurations
- UI routes and blocks
- Database settings
- Theming

## Project Structure

```
MICRO-ASXR/
├── server.js                 # Main server
├── runtime.html              # Single-page interface
├── package.json              # NPX configuration
├── MICRO.ASXR               # ASX configuration
├── bin/
│   └── cli.js               # CLI entry point
├── brain/
│   ├── bigrams.json         # N-gram data
│   └── trigrams.json        # N-gram data
├── python/
│   ├── asx_train.py         # Training script
│   ├── asx_infer.py         # Inference script
│   ├── asx_ngram_builder.py # N-gram builder
│   └── requirements.txt     # Python dependencies
├── models/
│   └── asx-gpt2/            # Transformer models
└── colab_training.ipynb     # Colab notebook
```

## GPU Support

### Enable GPU Runtime

```bash
# Set runtime to GPU
curl -X POST http://localhost:3000/api/runtime \
  -H "Content-Type: application/json" \
  -d '{"runtime": "gpu"}'
```

### Check GPU Status

```bash
# Check GPU availability
curl http://localhost:3000/api/runtime
```

### CUDA Setup

```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Start server with GPU
npm start
```

## WebSocket

Connect to `ws://localhost:3000` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:3000');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.channel === 'training') {
    console.log('Training progress:', data.data.progress);
  }
};

// Subscribe to updates
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'all'
}));
```

## Development

### Install Dependencies

```bash
npm install
```

### Python Dependencies

```bash
pip install -r python/requirements.txt
```

### Run Tests

```bash
npm test
```

### Development Mode

```bash
npm run dev
```

## Examples

### Example 1: Train and Predict

```javascript
// Train model
await fetch('http://localhost:3000/api/train/ngram', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    sentences: [
      'machine learning is amazing',
      'machine learning enables AI',
      'artificial intelligence is powerful'
    ]
  })
});

// Make prediction
const response = await fetch('http://localhost:3000/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'machine learning',
    maxResults: 3
  })
});

const predictions = await response.json();
console.log(predictions);
```

### Example 2: Load Custom Model

```javascript
// Load custom model
await fetch('http://localhost:3000/api/models/load', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    modelId: 'custom-ngram',
    type: 'ngram',
    path: 'brain/bigrams.json'
  })
});
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For issues and questions:
- GitHub Issues: https://github.com/cannaseedus-bot/MICRO-ASXR/issues
- Documentation: See this README

## Credits

Built by ASX-LABS

Powered by:
- Express.js
- PyTorch
- Transformers
- Node.js

---

**MX2LM MICRONAUT AI** - Your intelligent runtime companion