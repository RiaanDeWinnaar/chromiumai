/**
 * ChromiumAI Integration Header
 * Minimal interface for Chromium integration
 */

#ifndef CHROMIUM_AI_BROWSER_INTEGRATION_H_
#define CHROMIUM_AI_BROWSER_INTEGRATION_H_

namespace chromiumai {

/**
 * Initialize AI Browser integration
 * Called from chrome_browser_main.cc with minimal modification
 */
void InitializeAIBrowserIntegration();

/**
 * Register chrome://ai-browser/ protocol
 * Minimal WebUI registration
 */
void RegisterAIBrowserProtocol();

/**
 * Handle AI browser requests
 * Routes to our FastAPI service
 */
void HandleAIBrowserRequest(const std::string& url, std::string& response);

}  // namespace chromiumai

#endif  // CHROMIUM_AI_BROWSER_INTEGRATION_H_
