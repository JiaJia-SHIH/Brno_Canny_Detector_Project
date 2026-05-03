#include "server.hpp"
#include "threadpool.hpp"
#include "gaussian.hpp"
#include "gradient.hpp"
#include "nms.hpp"
#include "threshold.hpp"
#include "hysteresis.hpp"
#include "utils.hpp"

#include <opencv2/imgcodecs.hpp>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sstream>
#include <iostream>
#include <string>
#include <map>
#include <mutex>
#include <unordered_map>

using namespace std;

struct ResultCache {
    std::unordered_map<std::string, std::vector<uchar>> cache;
    std::mutex cacheMutex;
    
    std::string makeKey(int radius, float sigma, float lowR, float highR) {
        return std::to_string(radius) + "_" + std::to_string(sigma) + "_" + 
               std::to_string(lowR) + "_" + std::to_string(highR);
    }
    
    bool get(const std::string& key, std::vector<uchar>& result) {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = cache.find(key);
        if(it != cache.end()) {
            result = it->second;
            return true;
        }
        return false;
    }
    
    void set(const std::string& key, const std::vector<uchar>& data) {
        std::lock_guard<std::mutex> lock(cacheMutex);
        cache[key] = data;
    }
};

// Query string parsing
static std::map<std::string,int> parseQuery(const std::string& q)
{
    std::map<std::string,int> m;
    std::istringstream ss(q);
    std::string token;
    while(std::getline(ss, token, '&')) {
        auto eq = token.find('=');
        if(eq == std::string::npos) continue;
        std::string key = token.substr(0, eq);
        if(key == "t") continue;
        try {
            int val = std::stoi(token.substr(eq+1));
            m[key] = val;
        } catch(...) { continue; }
    }
    return m;
}

// utils: get the value from the map and set default value for it
static int get(const map<string, int>& m, const string& k, int def)
{
    auto it = m.find(k);
    return (it != m.end()) ? it->second : def;
}

// utils: combine two images
static cv::Mat hstack(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat out(a.rows, a.cols + b.cols, CV_8UC3, cv::Scalar(0));
    for(int i = 0; i < a.rows; i++)
    {
        for(int j = 0; j < a.cols; j++)
        {
            out.at<cv::Vec3b>(i, j) = a.at<cv::Vec3b>(i, j);
        }
        for(int j = 0; j < b.cols; j++)
        {
            out.at<cv::Vec3b>(i, j+a.cols) = b.at<cv::Vec3b>(i, j);
        }
    }
    return out;
}

static cv::Mat vstack(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat out(a.rows+b.rows, a.cols, CV_8UC3, cv::Scalar());
    for(int i = 0; i < a.rows; i++)
    {
        for(int j = 0; j < a.cols; j++)
        {
            out.at<cv::Vec3b>(i, j) = a.at<cv::Vec3b>(i, j);
        }
        for(int j = 0; j < b.cols; j++)
        {
            out.at<cv::Vec3b>(i+a.rows, j) = b.at<cv::Vec3b>(i, j);
        }   
    }
    return out;
}

static cv::Mat toColor(const cv::Mat& g8)
{
    return grayToBgrManually(g8);
}

static cv::Mat threshColor(const cv::Mat& t)
{
    cv::Mat out(t.size(), CV_8UC3);
    for(int i = 0; i < t.rows; i++)
    {
        for(int j = 0; j < t.cols; j++)
        {
            uchar v = t.at<uchar>(i, j);
            if(v == STRONG) out.at<cv::Vec3b>(i, j) = {255, 255, 255};
            else if(v == WEAK) out.at<cv::Vec3b>(i, j) = {0, 128, 255};
            else out.at<cv::Vec3b>(i, j) = {0, 0, 0};
        }
    }
    return out;
}

// full pipeline
static cv::Mat buildDisplay(const cv::Mat& src, int radius, float sigma, float lowR, float highR)
{
    radius = max(1, radius);
    if(lowR >= highR) lowR = highR * 0.5f;

    cv::Mat gray = bgrToGrayManually(src);
    cv::Mat blurred = gaussianBlur(gray, radius, sigma);
    GradientResult gradient = computeGradient(blurred);
    cv::Mat nms = nonMaximumSuppression(gradient);
    cv::Mat threshold = doubleThreshold(nms, lowR, highR);
    cv::Mat hysteresis = hysteresisTracking(threshold);

    cv::Mat panels[6] = 
    {
        toColor(normalizeManually(blurred)),
        toColor(normalizeManually(gradient.magnitude)),
        orientationToColorManually(gradient.orientation, normalizeManually(gradient.magnitude)),
        toColor(normalizeManually(nms)),
        threshColor(threshold),
        toColor(hysteresis)
    };

    cv::Mat row1 = hstack(hstack(panels[0], panels[1]), panels[2]);
    cv::Mat row2 = hstack(hstack(panels[3], panels[4]), panels[5]);

    return vstack(row1, row2);
}

// HTML 
static const std::string HTML = R"HTML(<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Canny Interactive</title>
<style>
  body { background:#1e1e1e; color:#ccc; font-family:monospace; padding:20px; }
  .ctrl { display:flex; align-items:center; gap:12px; margin:8px 0; }
  label { width:140px; text-align:right; }
  input[type=range] { width:300px; }
  span { width:40px; }
  img { margin-top:16px; max-width:100%; border:1px solid #444; }
</style>
</head>
<body>
<h2>Canny Edge Detector</h2>
<div class="ctrl"><label>Radius</label>
  <input type="range" id="radius" min="1" max="10" value="2">
  <span id="radius_v">2</span></div>
<div class="ctrl"><label>Sigma x10</label>
  <input type="range" id="sigma10" min="1" max="50" value="14">
  <span id="sigma10_v">1.4</span></div>
<div class="ctrl"><label>Low x100</label>
  <input type="range" id="low" min="1" max="40" value="5">
  <span id="low_v">0.05</span></div>
<div class="ctrl"><label>High x100</label>
  <input type="range" id="high" min="1" max="60" value="15">
  <span id="high_v">0.15</span></div>

<div style="color:#888;font-size:12px;margin-top:4px;">
  [Gaussian] [Magnitude] [Orientation] / [NMS] [Threshold] [Hysteresis]
</div>
<img id="result" src="/result?radius=2&sigma10=14&low=5&high=15">

<script>
let timer = null;
function update() {
  const r  = document.getElementById('radius').value;
  const s  = document.getElementById('sigma10').value;
  const lo = document.getElementById('low').value;
  const hi = document.getElementById('high').value;
  document.getElementById('radius_v').textContent = r;
  document.getElementById('sigma10_v').textContent = (s/10).toFixed(1);
  document.getElementById('low_v').textContent    = (lo/100).toFixed(2);
  document.getElementById('high_v').textContent   = (hi/100).toFixed(2);
  clearTimeout(timer);
  timer = setTimeout(() => {
    document.getElementById('result').src =
      `/result?radius=${r}&sigma10=${s}&low=${lo}&high=${hi}&t=${Date.now()}`;
  }, 80);  // debounce 80ms
}
['radius','sigma10','low','high'].forEach(id =>
  document.getElementById(id).addEventListener('input', update));
</script>
</body>
</html>)HTML";

// HTTP response 
static void sendResponse(int fd, const string& contentType, const char* body, size_t len)
{
    ostringstream hdr;
    hdr << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: " << contentType << "\r\n"
        << "Content-Length: " << len << "\r\n"
        << "Connection: close\r\n\r\n";
    string h = hdr.str();
    send(fd, h.c_str(), h.size(), 0);
    send(fd, body, len, 0);
}

// Main server loop
void runServer(const string& imgPath, int port)
{
    cv::Mat bgr = cv::imread(imgPath);
    if(bgr.empty())
    {
        cerr << "Cannot Load: " << imgPath << "\n";
        return;
    }

    cv::Mat src = bgrToGrayManually(bgr);
    ResultCache resultCache;

    ThreadPool pool(std::thread::hardware_concurrency());

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    bind(server_fd, (sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 5);

    cout << "Server running at http://localhost:" << port << "\n";
    cout << "Using " << std::thread::hardware_concurrency() << " worker threads\n";
    
    while(true)
    {
        int client = accept(server_fd, nullptr, nullptr);
        if(client < 0) continue;

        pool.enqueue([client, &src, &resultCache]() {
            char buffer[2048];
            recv(client, buffer, sizeof(buffer)-1, 0);
            string req(buffer);
            
            // get the path of the request
            string path;
            {
                auto start = req.find(' ');
                auto end = req.find(' ', start+1);
                if(start != string::npos && end != string::npos)
                {
                    path = req.substr(start+1, end-start-1);
                }
            }

            if(path == "/" || path.empty())
            {
                sendResponse(client, "text/html; charset=utf-8", HTML.c_str(), HTML.size());
            }
            else if(path.rfind("/result", 0) == 0)
            {
                string query;
                auto q = path.find('?');
                if(q != string::npos)
                {
                    query = path.substr(q+1);
                }
                auto params = parseQuery(query);

                int radius = get(params, "radius", 2);
                float sigma = get(params, "sigma10", 14) / 10.0f;
                float lowR = get(params, "low", 5) / 100.0f;
                float highR = get(params, "high", 15) / 100.0f;

                string cacheKey = resultCache.makeKey(radius, sigma, lowR, highR);
                vector<uchar> jpeg;
                
                if(!resultCache.get(cacheKey, jpeg))
                {
                    cv::Mat display = buildDisplay(src, radius, sigma, lowR, highR);
                    imencode(".jpg", display, jpeg, {cv::IMWRITE_JPEG_QUALITY, 90});
                    
                    resultCache.set(cacheKey, jpeg);
                }

                sendResponse(client, "image/jpeg", (const char*)jpeg.data(), jpeg.size());
            }

            close(client);
        });
    }

    close(server_fd);
    
}