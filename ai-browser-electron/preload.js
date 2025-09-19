/**
 * ChromiumAI Preload Script
 * Secure communication between renderer and main process
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose secure APIs to renderer process
contextBridge.exposeInMainWorld('chromiumAI', {
  // API communication methods
  async sendMessage(message) {
    return await ipcRenderer.invoke('send-message', message);
  },
  
  async getStatus() {
    return await ipcRenderer.invoke('get-status');
  },
  
  // WebUI protocol helpers
  async navigateTo(url) {
    return await ipcRenderer.invoke('navigate-to', url);
  },
  
  // ACP agent communication
  async requestAgentAction(agent, action, params) {
    return await ipcRenderer.invoke('agent-action', { agent, action, params });
  }
});

// Listen for updates from main process
ipcRenderer.on('status-update', (event, status) => {
  if (window.chromiumAI && window.chromiumAI.onStatusUpdate) {
    window.chromiumAI.onStatusUpdate(status);
  }
});

ipcRenderer.on('agent-response', (event, response) => {
  if (window.chromiumAI && window.chromiumAI.onAgentResponse) {
    window.chromiumAI.onAgentResponse(response);
  }
});
