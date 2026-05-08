/**
 * @file server.hpp
 * @author SHIH YUE JIA (xshihyu00)
 * @brief HTTP server for interactive Canny parameter tuning
 **/

#pragma once
#include <string>
#include <opencv2/core.hpp>

// run server
void runServer(const std::string& imgPath, int port = 8080);