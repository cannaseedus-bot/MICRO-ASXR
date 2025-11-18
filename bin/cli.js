#!/usr/bin/env node
/**
 * ASXR CLI - NPX entry point
 * Run: npx @asxr/runtime-server
 * Or: asxr [command] [options]
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const args = process.argv.slice(2);
const command = args[0] || 'start';

const commands = {
  start: {
    description: 'Start the ASXR runtime server',
    action: () => {
      console.log('🚀 Starting ASXR Runtime Server...\n');
      const serverPath = path.join(__dirname, '..', 'server.js');
      const server = spawn('node', [serverPath], {
        stdio: 'inherit',
        cwd: path.join(__dirname, '..')
      });

      server.on('error', (error) => {
        console.error('Failed to start server:', error);
        process.exit(1);
      });

      process.on('SIGINT', () => {
        server.kill('SIGINT');
        process.exit(0);
      });
    }
  },

  serve: {
    description: 'Alias for start',
    action: () => commands.start.action()
  },

  dev: {
    description: 'Start in development mode with auto-reload',
    action: () => {
      console.log('🔧 Starting ASXR in development mode...\n');
      const nodemonPath = require.resolve('nodemon/bin/nodemon.js');
      const serverPath = path.join(__dirname, '..', 'server.js');
      const dev = spawn('node', [nodemonPath, serverPath], {
        stdio: 'inherit',
        cwd: path.join(__dirname, '..')
      });

      dev.on('error', (error) => {
        console.error('Failed to start dev server:', error);
        process.exit(1);
      });

      process.on('SIGINT', () => {
        dev.kill('SIGINT');
        process.exit(0);
      });
    }
  },

  train: {
    description: 'Train a model with a dataset',
    action: () => {
      const datasetPath = args[1];

      if (!datasetPath) {
        console.error('Usage: asxr train <dataset-path>');
        process.exit(1);
      }

      console.log(`🎓 Training model with dataset: ${datasetPath}\n`);
      const trainScript = path.join(__dirname, '..', 'python', 'asx_train.py');
      const train = spawn('python3', [trainScript, datasetPath], {
        stdio: 'inherit'
      });

      train.on('close', (code) => {
        if (code === 0) {
          console.log('\n✓ Training completed successfully');
        } else {
          console.error(`\n✗ Training failed with code ${code}`);
          process.exit(code);
        }
      });
    }
  },

  predict: {
    description: 'Make a prediction with the model',
    action: () => {
      const text = args.slice(1).join(' ');

      if (!text) {
        console.error('Usage: asxr predict <text>');
        process.exit(1);
      }

      console.log(`🔮 Predicting next word for: "${text}"\n`);
      const inferScript = path.join(__dirname, '..', 'python', 'asx_infer.py');
      const infer = spawn('python3', [inferScript, text], {
        stdio: 'inherit'
      });

      infer.on('close', (code) => {
        if (code !== 0) {
          console.error(`\n✗ Prediction failed with code ${code}`);
          process.exit(code);
        }
      });
    }
  },

  init: {
    description: 'Initialize a new ASXR project',
    action: () => {
      const projectName = args[1] || 'my-asxr-project';
      const projectPath = path.join(process.cwd(), projectName);

      console.log(`📦 Initializing ASXR project: ${projectName}\n`);

      try {
        if (fs.existsSync(projectPath)) {
          console.error(`Error: Directory ${projectName} already exists`);
          process.exit(1);
        }

        fs.mkdirSync(projectPath, { recursive: true });
        fs.mkdirSync(path.join(projectPath, 'brain'), { recursive: true });
        fs.mkdirSync(path.join(projectPath, 'models'), { recursive: true });
        fs.mkdirSync(path.join(projectPath, 'datasets'), { recursive: true });

        // Copy template files
        const templateConfig = {
          asxr_version: 2,
          meta: {
            name: projectName,
            version: "1.0.0"
          }
        };

        fs.writeFileSync(
          path.join(projectPath, 'MICRO.ASXR'),
          JSON.stringify(templateConfig, null, 2)
        );

        fs.writeFileSync(
          path.join(projectPath, 'brain', 'bigrams.json'),
          '{}'
        );

        fs.writeFileSync(
          path.join(projectPath, 'brain', 'trigrams.json'),
          '{}'
        );

        console.log(`✓ Project created at: ${projectPath}`);
        console.log(`\nNext steps:`);
        console.log(`  cd ${projectName}`);
        console.log(`  npx @asxr/runtime-server start`);
      } catch (error) {
        console.error('Failed to initialize project:', error.message);
        process.exit(1);
      }
    }
  },

  version: {
    description: 'Show version information',
    action: () => {
      const packagePath = path.join(__dirname, '..', 'package.json');
      const pkg = require(packagePath);
      console.log(`ASXR Runtime Server v${pkg.version}`);
    }
  },

  help: {
    description: 'Show help information',
    action: () => {
      console.log('ASXR Runtime Server - NPX CLI\n');
      console.log('Usage: asxr [command] [options]\n');
      console.log('Commands:');

      Object.entries(commands).forEach(([name, cmd]) => {
        console.log(`  ${name.padEnd(12)} ${cmd.description}`);
      });

      console.log('\nExamples:');
      console.log('  asxr start                    # Start the server');
      console.log('  asxr dev                      # Start in dev mode');
      console.log('  asxr train dataset.txt        # Train a model');
      console.log('  asxr predict "hello world"    # Make prediction');
      console.log('  asxr init my-project          # Create new project');
      console.log('\nEnvironment Variables:');
      console.log('  PORT=3000                     # Server port (default: 3000)');
      console.log('  HOST=0.0.0.0                  # Server host (default: 0.0.0.0)');
      console.log('  CUDA_VISIBLE_DEVICES=0        # GPU device (for GPU runtime)');
    }
  }
};

// Execute command
if (commands[command]) {
  commands[command].action();
} else {
  console.error(`Unknown command: ${command}`);
  console.log('Run "asxr help" for usage information');
  process.exit(1);
}
