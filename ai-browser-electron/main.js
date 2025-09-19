/**
 * ChromiumAI Electron Main Process
 * Implements chrome://ai-browser/ protocol using Electron
 */

const { app, BrowserWindow, protocol, ipcMain } = require('electron');
const path = require('path');
const http = require('http');

// FastAPI service configuration
const API_BASE_URL = 'http://localhost:3456';

class ChromiumAI {
  constructor() {
    this.mainWindow = null;
    this.apiService = null;
  }

  async createWindow() {
    // Create the browser window
    this.mainWindow = new BrowserWindow({
      width: 1400,
      height: 900,
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        enableRemoteModule: false,
        preload: path.join(__dirname, 'preload.js')
      },
      icon: path.join(__dirname, 'assets/icon.png')
    });

    // Register custom protocol
    this.registerAIBrowserProtocol();

    // Load the main page
    await this.mainWindow.loadURL('chrome://ai-browser/');

    // Open DevTools in development
    if (process.env.NODE_ENV === 'development') {
      this.mainWindow.webContents.openDevTools();
    }
  }

  registerAIBrowserProtocol() {
    // Register chrome://ai-browser/ protocol
    protocol.registerHttpProtocol('chrome', (request, callback) => {
      const url = new URL(request.url);
      
      if (url.hostname === 'ai-browser') {
        // Handle chrome://ai-browser/ requests
        this.handleAIBrowserRequest(request, callback);
      } else {
        // Let other chrome:// protocols pass through
        callback({ url: request.url });
      }
    });

    // Register file protocol for local resources
    protocol.registerFileProtocol('chrome', (request, callback) => {
      const url = new URL(request.url);
      
      if (url.hostname === 'ai-browser') {
        const filePath = path.join(__dirname, 'webui', url.pathname);
        callback({ path: filePath });
      } else {
        callback({ error: -6 }); // FILE_NOT_FOUND
      }
    });
  }

  async handleAIBrowserRequest(request, callback) {
    const url = new URL(request.url);
    const pathname = url.pathname;

    try {
      if (pathname === '/' || pathname === '/index.html') {
        // Serve the main AI browser interface
        const html = await this.generateAIBrowserHTML();
        callback({
          mimeType: 'text/html',
          data: html
        });
      } else if (pathname.startsWith('/api/')) {
        // Proxy API requests to FastAPI service
        await this.proxyAPIRequest(request, callback);
      } else {
        // Serve static files
        const filePath = path.join(__dirname, 'webui', pathname);
        callback({ path: filePath });
      }
    } catch (error) {
      console.error('Error handling AI browser request:', error);
      callback({
        mimeType: 'text/html',
        data: this.generateErrorHTML(error.message)
      });
    }
  }

  async generateAIBrowserHTML() {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChromiumAI - AI-Native Browser</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header h1 {
            font-size: 3rem;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
            margin: 10px 0;
        }
        .ai-interface {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .chat-container {
            height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            background: rgba(0,0,0,0.2);
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 80%;
        }
        .user-message {
            background: rgba(255,255,255,0.2);
            margin-left: auto;
            text-align: right;
        }
        .ai-message {
            background: rgba(0,255,255,0.2);
            margin-right: auto;
        }
        .input-container {
            display: flex;
            gap: 10px;
        }
        .input-container input {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 25px;
            background: rgba(255,255,255,0.2);
            color: white;
            font-size: 16px;
        }
        .input-container input::placeholder {
            color: rgba(255,255,255,0.7);
        }
        .input-container button {
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            background: #4CAF50;
            color: white;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .input-container button:hover {
            background: #45a049;
        }
        .status {
            text-align: center;
            margin-top: 20px;
            opacity: 0.8;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ChromiumAI</h1>
            <p>AI-Native Browser with Invisible Integration</p>
            <p>Powered by ACP Agent Swarm & GAIA Optimization</p>
        </div>
        
        <div class="ai-interface">
            <div class="chat-container" id="chatContainer">
                <div class="message ai-message">
                    <strong>ChromiumAI:</strong> Welcome! I'm your AI-native browser assistant. How can I help you today?
                </div>
            </div>
            
            <div class="input-container">
                <input type="text" id="userInput" placeholder="Ask me anything or request a web action..." />
                <button onclick="sendMessage()">Send</button>
            </div>
            
            <div class="status" id="status">
                Status: Connected to ACP Runtime
            </div>
        </div>
    </div>

    <script>
        const API_BASE = 'http://localhost:3456';
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';
            
            // Update status
            updateStatus('Processing with ACP agents...');
            
            try {
                // Send to FastAPI service
                const response = await fetch(\`\${API_BASE}/api/chat\`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        context: 'ai-browser'
                    })
                });
                
                const data = await response.json();
                
                // Add AI response to chat
                addMessage(data.response, 'ai');
                updateStatus('Ready');
                
            } catch (error) {
                console.error('Error:', error);
                addMessage('Sorry, I encountered an error. Please try again.', 'ai');
                updateStatus('Error - Check FastAPI service');
            }
        }
        
        function addMessage(text, type) {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = \`message \${type}-message\`;
            messageDiv.innerHTML = \`<strong>\${type === 'user' ? 'You' : 'ChromiumAI'}:</strong> \${text}\`;
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }
        
        function updateStatus(text) {
            document.getElementById('status').textContent = \`Status: \${text}\`;
        }
        
        // Allow Enter key to send message
        document.getElementById('userInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Check API connection on load
        window.addEventListener('load', async () => {
            try {
                const response = await fetch(\`\${API_BASE}/health\`);
                if (response.ok) {
                    updateStatus('Connected to ACP Runtime');
                } else {
                    updateStatus('API service not responding');
                }
            } catch (error) {
                updateStatus('API service offline - Start FastAPI service');
            }
        });
    </script>
</body>
</html>`;
  }

  generateErrorHTML(error) {
    return `
<!DOCTYPE html>
<html>
<head>
    <title>ChromiumAI Error</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }
        .error { background: #ffebee; border: 1px solid #f44336; padding: 20px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="error">
        <h2>ChromiumAI Error</h2>
        <p>${error}</p>
    </div>
</body>
</html>`;
  }

  async proxyAPIRequest(request, callback) {
    // Proxy API requests to FastAPI service
    const apiPath = request.url.replace('chrome://ai-browser/api', API_BASE_URL + '/api');
    
    // Make request to FastAPI service
    const options = {
      hostname: 'localhost',
      port: 3456,
      path: request.url.replace('chrome://ai-browser', ''),
      method: request.method,
      headers: request.headers
    };

    const proxyReq = http.request(options, (proxyRes) => {
      let data = '';
      proxyRes.on('data', chunk => data += chunk);
      proxyRes.on('end', () => {
        callback({
          mimeType: proxyRes.headers['content-type'] || 'application/json',
          data: data
        });
      });
    });

    proxyReq.on('error', (error) => {
      callback({
        mimeType: 'application/json',
        data: JSON.stringify({ error: 'API service unavailable' })
      });
    });

    proxyReq.end();
  }
}

// Initialize ChromiumAI
const chromiumAI = new ChromiumAI();

// App event handlers
app.whenReady().then(() => {
  chromiumAI.createWindow();
  
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      chromiumAI.createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Handle protocol registration
app.setAsDefaultProtocolClient('chrome');
