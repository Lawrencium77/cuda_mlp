#include "config_reader.h"

std::map<std::string, std::string> read_config(const std::string& filename) {
    std::map<std::string, std::string> config;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream line_stream(line);
        std::string key;
        if (std::getline(line_stream, key, '=')) {
            std::string value;
            if (std::getline(line_stream, value)) {
                config[key] = value;
            }
        }
    }
    return config;
}
