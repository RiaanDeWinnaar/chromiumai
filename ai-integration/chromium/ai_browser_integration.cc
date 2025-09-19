/**
 * ChromiumAI Integration Implementation
 * Minimal Chromium integration - keeps our code separate
 */

#include "ai_browser_integration.h"
#include "base/logging.h"
#include "chrome/browser/profiles/profile.h"
#include "chrome/browser/ui/browser.h"
#include "chrome/browser/ui/browser_list.h"
#include "chrome/browser/ui/tabs/tab_strip_model.h"
#include "content/public/browser/web_contents.h"
#include "content/public/browser/web_ui.h"
#include "content/public/browser/web_ui_controller.h"
#include "content/public/browser/web_ui_data_source.h"
#include "content/public/browser/web_ui_message_handler.h"
#include "net/http/http_request_headers.h"
#include "net/http/http_response_headers.h"
#include "net/url_request/url_request.h"
#include "net/url_request/url_request_job.h"
#include "net/url_request/url_request_job_factory.h"
#include "net/url_request/url_request_interceptor.h"
#include "url/gurl.h"

#include <memory>
#include <string>

namespace chromiumai {

// FastAPI service configuration
const char kAIBrowserAPIBase[] = "http://localhost:3456";

class AIBrowserWebUIController : public content::WebUIController {
 public:
  AIBrowserWebUIController(content::WebUI* web_ui);
  ~AIBrowserWebUIController() override;
};

class AIBrowserWebUIDataSource : public content::WebUIDataSource {
 public:
  AIBrowserWebUIDataSource();
  ~AIBrowserWebUIDataSource() override;

 private:
  void StartDataRequest(const GURL& url,
                       const content::WebContents::Getter& wc_getter,
                       content::URLDataSource::GotDataCallback callback) override;
};

class AIBrowserURLRequestInterceptor : public net::URLRequestInterceptor {
 public:
  AIBrowserURLRequestInterceptor();
  ~AIBrowserURLRequestInterceptor() override;

  net::URLRequestJob* MaybeInterceptRequest(
      net::URLRequest* request,
      net::NetworkDelegate* network_delegate) const override;
};

void InitializeAIBrowserIntegration() {
  LOG(INFO) << "ChromiumAI: Initializing AI Browser integration";
  
  // Register chrome://ai-browser/ protocol
  RegisterAIBrowserProtocol();
  
  LOG(INFO) << "ChromiumAI: AI Browser integration initialized";
}

void RegisterAIBrowserProtocol() {
  // Register WebUI data source for chrome://ai-browser/
  content::WebUIDataSource* source = new AIBrowserWebUIDataSource();
  content::WebUIDataSource::Add(Profile::GetLastUsedProfile(), source);
  
  LOG(INFO) << "ChromiumAI: Registered chrome://ai-browser/ protocol";
}

void HandleAIBrowserRequest(const std::string& url, std::string& response) {
  // Minimal implementation - route to FastAPI service
  // This keeps our AI logic separate from Chromium
  response = "AI Browser request handled by external service";
}

// WebUI Controller Implementation
AIBrowserWebUIController::AIBrowserWebUIController(content::WebUI* web_ui)
    : WebUIController(web_ui) {
  web_ui->AddMessageHandler(std::make_unique<AIBrowserWebUIDataSource>());
}

AIBrowserWebUIController::~AIBrowserWebUIController() = default;

// WebUI Data Source Implementation
AIBrowserWebUIDataSource::AIBrowserWebUIDataSource() {
  SetRequestFilter(base::BindRepeating([](const std::string& path) {
    return path == "ai-browser" || path == "ai-browser/";
  }));
}

AIBrowserWebUIDataSource::~AIBrowserWebUIDataSource() = default;

void AIBrowserWebUIDataSource::StartDataRequest(
    const GURL& url,
    const content::WebContents::Getter& wc_getter,
    content::URLDataSource::GotDataCallback callback) {
  
  std::string response;
  HandleAIBrowserRequest(url.spec(), response);
  
  std::move(callback).Run(base::MakeRefCounted<base::RefCountedString>(response));
}

// URL Request Interceptor Implementation
AIBrowserURLRequestInterceptor::AIBrowserURLRequestInterceptor() = default;
AIBrowserURLRequestInterceptor::~AIBrowserURLRequestInterceptor() = default;

net::URLRequestJob* AIBrowserURLRequestInterceptor::MaybeInterceptRequest(
    net::URLRequest* request,
    net::NetworkDelegate* network_delegate) const {
  
  if (request->url().scheme() == "chrome" && 
      request->url().host() == "ai-browser") {
    // Handle chrome://ai-browser/ requests
    return nullptr; // Let WebUI handle it
  }
  
  return nullptr;
}

}  // namespace chromiumai
