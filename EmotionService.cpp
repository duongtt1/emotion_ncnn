#include "EmotionService.h"

EmotionService *EmotionService::m_instance = 0;
std::mutex EmotionService::m_ctx;

EmotionService* EmotionService::getInstance()
{
    std::unique_lock<std::mutex> lock(m_ctx);
        if (!m_instance)
            m_instance = new EmotionService();
        return m_instance;
}

bool EmotionService::getLabels(std::string file_path){
    std::string line;
    bool ret = true;
    std::ifstream file(file_path);
    try {
        if (file.is_open()) {
            while (std::getline(file, line)) {
                if (!line.empty())
                    this->vt_label.push_back(line);
            }
            file.close();
        } else {
            throw std::runtime_error("Error: Unable to open file '" + file_path + "'");
        }
    }catch (const std::exception& e) {
        ret = false;
        std::cerr << e.what() << std::endl;
    }
    return ret;
}

bool EmotionService::init_models(   std::string path_file_param, 
                                    std::string path_file_bin   )
{
    bool ret = true;
    try
    {
        this->mobilenet.load_param(path_file_param);
        this->mobilenet.load_model(path_file_bin);
        this->mobilenet.substract_mean_normalize(mean_vals, norm_vals);
    }
    catch(const std::exception& e)
    {
        ret = false;
        cerr << "Error loading model: " << e.what() << endl;
    }
    
    return ret;
}

std::string EmotionService::predict(const cv::Mat& img){
    
} 

EmotionService::EmotionService(/* args */)
{
}

EmotionService::~EmotionService()
{
}