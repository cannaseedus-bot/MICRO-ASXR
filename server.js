#!/usr/bin/env node
/**
 * MICRO-ASXR Runtime Server
 * Single-page runtime with REST API and NPX support
 * Supports CPU/GPU backends and Colab training integration
 */

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const compression = require('compression');
const helmet = require('helmet');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');
const WebSocket = require('ws');
const http = require('http');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || '0.0.0.0';

// Create HTTP server for WebSocket support
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Middleware
app.use(helmet({
  contentSecurityPolicy: false, // Allow inline scripts for runtime
}));
app.use(cors());
app.use(compression());
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '50mb' }));
app.use(morgan('dev'));

// Serve static files
app.use(express.static(__dirname));

// Runtime state
const state = {
  asxConfig: null,
  models: new Map(),
  trainingJobs: new Map(),
  runtime: 'cpu', // 'cpu' or 'gpu'
  status: 'idle',
  stats: {
    requests: 0,
    predictions: 0,
    trainings: 0,
    uptime: Date.now()
  }
};

// Load ASX configuration
async function loadASXConfig() {
  try {
    const asxPath = path.join(__dirname, 'MICRO.ASXR');
    const content = await fs.readFile(asxPath, 'utf8');
    state.asxConfig = JSON.parse(content);
    console.log('✓ ASX Configuration loaded');
    return state.asxConfig;
  } catch (error) {
    console.error('Failed to load ASX config:', error.message);
    return null;
  }
}

// Initialize brain data
async function initBrain() {
  const brainPath = path.join(__dirname, 'brain');
  try {
    await fs.mkdir(brainPath, { recursive: true });

    // Create default brain files if they don't exist
    const bigramsPath = path.join(brainPath, 'bigrams.json');
    const trigramsPath = path.join(brainPath, 'trigrams.json');

    try {
      await fs.access(bigramsPath);
    } catch {
      await fs.writeFile(bigramsPath, JSON.stringify({}, null, 2));
    }

    try {
      await fs.access(trigramsPath);
    } catch {
      await fs.writeFile(trigramsPath, JSON.stringify({}, null, 2));
    }

    console.log('✓ Brain initialized');
  } catch (error) {
    console.error('Failed to initialize brain:', error.message);
  }
}

// ============================================================================
// REST API ENDPOINTS
// ============================================================================

// Health check
app.get('/api/health', (req, res) => {
  state.stats.requests++;
  res.json({
    status: 'ok',
    runtime: state.runtime,
    uptime: Date.now() - state.stats.uptime,
    stats: state.stats
  });
});

// Get system info
app.get('/api/system', async (req, res) => {
  state.stats.requests++;
  res.json({
    version: state.asxConfig?.meta?.version || 'unknown',
    serial: state.asxConfig?.meta?.serial || 'unknown',
    runtime: state.runtime,
    models: Array.from(state.models.keys()),
    stats: state.stats
  });
});

// Get ASX configuration
app.get('/api/config', (req, res) => {
  state.stats.requests++;
  res.json(state.asxConfig || {});
});

// Update ASX configuration
app.post('/api/config', async (req, res) => {
  try {
    const newConfig = req.body;
    const asxPath = path.join(__dirname, 'MICRO.ASXR');
    await fs.writeFile(asxPath, JSON.stringify(newConfig, null, 2));
    state.asxConfig = newConfig;
    res.json({ success: true, config: newConfig });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ============================================================================
// MODEL ENDPOINTS
// ============================================================================

// List models
app.get('/api/models', (req, res) => {
  state.stats.requests++;
  const models = Array.from(state.models.entries()).map(([id, model]) => ({
    id,
    ...model
  }));
  res.json({ models });
});

// Load model
app.post('/api/models/load', async (req, res) => {
  try {
    const { modelId, type, path: modelPath } = req.body;

    if (!modelId || !type) {
      return res.status(400).json({ error: 'modelId and type required' });
    }

    // Load model data
    let modelData = {};
    if (modelPath) {
      const fullPath = path.join(__dirname, modelPath);
      const content = await fs.readFile(fullPath, 'utf8');
      modelData = JSON.parse(content);
    }

    state.models.set(modelId, {
      id: modelId,
      type,
      path: modelPath,
      data: modelData,
      loaded: Date.now()
    });

    res.json({ success: true, modelId });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Unload model
app.delete('/api/models/:modelId', (req, res) => {
  const { modelId } = req.params;

  if (state.models.has(modelId)) {
    state.models.delete(modelId);
    res.json({ success: true, modelId });
  } else {
    res.status(404).json({ error: 'Model not found' });
  }
});

// ============================================================================
// PREDICTION/INFERENCE ENDPOINTS
// ============================================================================

// N-gram prediction
app.post('/api/predict', async (req, res) => {
  try {
    state.stats.requests++;
    state.stats.predictions++;

    const { text, modelId = 'ngram', maxResults = 5 } = req.body;

    if (!text) {
      return res.status(400).json({ error: 'text required' });
    }

    // Load brain data
    const bigramsPath = path.join(__dirname, 'brain', 'bigrams.json');
    const trigramsPath = path.join(__dirname, 'brain', 'trigrams.json');

    let bigrams = {};
    let trigrams = {};

    try {
      const bigramsContent = await fs.readFile(bigramsPath, 'utf8');
      bigrams = JSON.parse(bigramsContent);
    } catch (e) {}

    try {
      const trigramsContent = await fs.readFile(trigramsPath, 'utf8');
      trigrams = JSON.parse(trigramsContent);
    } catch (e) {}

    // Simple n-gram prediction
    const tokens = text.toLowerCase().split(/\s+/);
    const predictions = [];

    if (tokens.length >= 2) {
      const key = tokens.slice(-2).join(' ');
      if (trigrams[key]) {
        predictions.push(...Object.entries(trigrams[key])
          .sort((a, b) => b[1] - a[1])
          .slice(0, maxResults)
          .map(([word, count]) => ({ word, score: count })));
      }
    }

    if (predictions.length < maxResults && tokens.length >= 1) {
      const key = tokens[tokens.length - 1];
      if (bigrams[key]) {
        const remaining = maxResults - predictions.length;
        predictions.push(...Object.entries(bigrams[key])
          .sort((a, b) => b[1] - a[1])
          .slice(0, remaining)
          .map(([word, count]) => ({ word, score: count })));
      }
    }

    res.json({
      input: text,
      predictions: predictions.length > 0 ? predictions : [{ word: 'the', score: 1 }],
      modelId
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Python inference (for more complex models)
app.post('/api/infer', async (req, res) => {
  try {
    state.stats.requests++;
    state.stats.predictions++;

    const { text, modelPath, runtime = state.runtime } = req.body;

    if (!text) {
      return res.status(400).json({ error: 'text required' });
    }

    // Call Python inference script
    const pythonScript = path.join(__dirname, 'python', 'asx_infer.py');
    const args = [text];

    if (modelPath) args.push('--model', modelPath);
    if (runtime === 'gpu') args.push('--gpu');

    const result = await runPythonScript(pythonScript, args);

    res.json({
      input: text,
      result: result,
      runtime
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ============================================================================
// TRAINING ENDPOINTS
// ============================================================================

// Train n-gram model
app.post('/api/train/ngram', async (req, res) => {
  try {
    state.stats.requests++;
    state.stats.trainings++;

    const { text, sentences } = req.body;

    if (!text && !sentences) {
      return res.status(400).json({ error: 'text or sentences required' });
    }

    const trainingData = sentences || [text];

    // Load existing brain
    const bigramsPath = path.join(__dirname, 'brain', 'bigrams.json');
    const trigramsPath = path.join(__dirname, 'brain', 'trigrams.json');

    let bigrams = {};
    let trigrams = {};

    try {
      const bigramsContent = await fs.readFile(bigramsPath, 'utf8');
      bigrams = JSON.parse(bigramsContent);
    } catch (e) {}

    try {
      const trigramsContent = await fs.readFile(trigramsPath, 'utf8');
      trigrams = JSON.parse(trigramsContent);
    } catch (e) {}

    // Train on data
    for (const sentence of trainingData) {
      const tokens = sentence.toLowerCase().split(/\s+/);

      // Build bigrams
      for (let i = 0; i < tokens.length - 1; i++) {
        const key = tokens[i];
        const next = tokens[i + 1];

        if (!bigrams[key]) bigrams[key] = {};
        bigrams[key][next] = (bigrams[key][next] || 0) + 1;
      }

      // Build trigrams
      for (let i = 0; i < tokens.length - 2; i++) {
        const key = `${tokens[i]} ${tokens[i + 1]}`;
        const next = tokens[i + 2];

        if (!trigrams[key]) trigrams[key] = {};
        trigrams[key][next] = (trigrams[key][next] || 0) + 1;
      }
    }

    // Save updated brain
    await fs.writeFile(bigramsPath, JSON.stringify(bigrams, null, 2));
    await fs.writeFile(trigramsPath, JSON.stringify(trigrams, null, 2));

    res.json({
      success: true,
      sentencesTrained: trainingData.length,
      bigramKeys: Object.keys(bigrams).length,
      trigramKeys: Object.keys(trigrams).length
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Start training job
app.post('/api/train/start', async (req, res) => {
  try {
    state.stats.trainings++;

    const {
      dataset,
      modelType = 'ngram',
      runtime = state.runtime,
      epochs = 10,
      batchSize = 32
    } = req.body;

    if (!dataset) {
      return res.status(400).json({ error: 'dataset required' });
    }

    const jobId = `job_${Date.now()}`;

    state.trainingJobs.set(jobId, {
      id: jobId,
      status: 'running',
      modelType,
      runtime,
      epochs,
      batchSize,
      started: Date.now(),
      progress: 0
    });

    // Start training in background
    trainModelAsync(jobId, dataset, modelType, runtime, epochs, batchSize)
      .catch(error => {
        const job = state.trainingJobs.get(jobId);
        if (job) {
          job.status = 'failed';
          job.error = error.message;
        }
      });

    res.json({
      success: true,
      jobId,
      status: 'running'
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get training job status
app.get('/api/train/status/:jobId', (req, res) => {
  const { jobId } = req.params;
  const job = state.trainingJobs.get(jobId);

  if (job) {
    res.json(job);
  } else {
    res.status(404).json({ error: 'Job not found' });
  }
});

// List training jobs
app.get('/api/train/jobs', (req, res) => {
  const jobs = Array.from(state.trainingJobs.values());
  res.json({ jobs });
});

// ============================================================================
// RUNTIME CONTROL ENDPOINTS
// ============================================================================

// Set runtime (CPU/GPU)
app.post('/api/runtime', (req, res) => {
  const { runtime } = req.body;

  if (runtime !== 'cpu' && runtime !== 'gpu') {
    return res.status(400).json({ error: 'runtime must be "cpu" or "gpu"' });
  }

  state.runtime = runtime;
  res.json({ success: true, runtime: state.runtime });
});

// Get runtime info
app.get('/api/runtime', (req, res) => {
  res.json({
    runtime: state.runtime,
    available: ['cpu', 'gpu'],
    gpu: {
      available: process.env.CUDA_VISIBLE_DEVICES !== undefined,
      device: process.env.CUDA_VISIBLE_DEVICES || 'none'
    }
  });
});

// ============================================================================
// FILESYSTEM/BRAIN ENDPOINTS
// ============================================================================

// Get brain data
app.get('/api/brain/:type', async (req, res) => {
  try {
    const { type } = req.params;

    if (type !== 'bigrams' && type !== 'trigrams') {
      return res.status(400).json({ error: 'type must be "bigrams" or "trigrams"' });
    }

    const brainPath = path.join(__dirname, 'brain', `${type}.json`);
    const content = await fs.readFile(brainPath, 'utf8');
    const data = JSON.parse(content);

    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Update brain data
app.post('/api/brain/:type', async (req, res) => {
  try {
    const { type } = req.params;
    const data = req.body;

    if (type !== 'bigrams' && type !== 'trigrams') {
      return res.status(400).json({ error: 'type must be "bigrams" or "trigrams"' });
    }

    const brainPath = path.join(__dirname, 'brain', `${type}.json`);
    await fs.writeFile(brainPath, JSON.stringify(data, null, 2));

    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ============================================================================
// WEBSOCKET FOR REAL-TIME UPDATES
// ============================================================================

wss.on('connection', (ws) => {
  console.log('WebSocket client connected');

  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);

      // Handle different message types
      if (data.type === 'subscribe') {
        ws.subscribed = data.channel || 'all';
      }
    } catch (error) {
      ws.send(JSON.stringify({ error: error.message }));
    }
  });

  ws.on('close', () => {
    console.log('WebSocket client disconnected');
  });

  // Send initial status
  ws.send(JSON.stringify({
    type: 'status',
    data: {
      runtime: state.runtime,
      models: Array.from(state.models.keys()),
      stats: state.stats
    }
  }));
});

// Broadcast to all WebSocket clients
function broadcast(channel, data) {
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      if (!client.subscribed || client.subscribed === 'all' || client.subscribed === channel) {
        client.send(JSON.stringify({ channel, data }));
      }
    }
  });
}

// ============================================================================
// SERVE SINGLE-PAGE RUNTIME
// ============================================================================

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'runtime.html'));
});

// Fallback to runtime for SPA routing
app.get('*', (req, res) => {
  if (!req.path.startsWith('/api/')) {
    res.sendFile(path.join(__dirname, 'runtime.html'));
  } else {
    res.status(404).json({ error: 'Endpoint not found' });
  }
});

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Run Python script
function runPythonScript(scriptPath, args = []) {
  return new Promise((resolve, reject) => {
    const python = spawn('python3', [scriptPath, ...args]);

    let output = '';
    let error = '';

    python.stdout.on('data', (data) => {
      output += data.toString();
    });

    python.stderr.on('data', (data) => {
      error += data.toString();
    });

    python.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(output));
        } catch {
          resolve(output.trim());
        }
      } else {
        reject(new Error(error || `Process exited with code ${code}`));
      }
    });
  });
}

// Async training function
async function trainModelAsync(jobId, dataset, modelType, runtime, epochs, batchSize) {
  const job = state.trainingJobs.get(jobId);

  try {
    // Simulate training progress
    for (let epoch = 0; epoch < epochs; epoch++) {
      await new Promise(resolve => setTimeout(resolve, 1000));

      job.progress = ((epoch + 1) / epochs) * 100;
      job.epoch = epoch + 1;

      // Broadcast progress
      broadcast('training', {
        jobId,
        progress: job.progress,
        epoch: job.epoch
      });
    }

    job.status = 'completed';
    job.completed = Date.now();

    broadcast('training', {
      jobId,
      status: 'completed'
    });
  } catch (error) {
    job.status = 'failed';
    job.error = error.message;
    throw error;
  }
}

// ============================================================================
// STARTUP
// ============================================================================

async function startup() {
  console.log('\n🚀 MICRO-ASXR Runtime Server');
  console.log('================================\n');

  // Load configuration
  await loadASXConfig();
  await initBrain();

  // Start server
  server.listen(PORT, HOST, () => {
    console.log(`\n✓ Server running on http://${HOST}:${PORT}`);
    console.log(`✓ Runtime mode: ${state.runtime.toUpperCase()}`);
    console.log(`✓ WebSocket: ws://${HOST}:${PORT}`);
    console.log('\nAPI Endpoints:');
    console.log('  GET  /api/health          - Health check');
    console.log('  GET  /api/system          - System info');
    console.log('  GET  /api/config          - Get ASX config');
    console.log('  POST /api/predict         - N-gram prediction');
    console.log('  POST /api/train/ngram     - Train n-gram model');
    console.log('  POST /api/train/start     - Start training job');
    console.log('  POST /api/runtime         - Set runtime (cpu/gpu)');
    console.log('  GET  /api/models          - List models');
    console.log('\n  Visit http://localhost:' + PORT + ' for the runtime interface\n');
  });
}

// Error handling
process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
});

process.on('unhandledRejection', (error) => {
  console.error('Unhandled rejection:', error);
});

// Start the server
startup();

module.exports = { app, server };
