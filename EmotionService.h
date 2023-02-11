#include <iostream>
#include <mutex>
#include <fstream>
#include <vector>
#include <string>

#include "ncnn/net.h"


class EmotionService
{
private:
    static EmotionService *m_instance;
    static std::mutex m_ctx;
    EmotionService() {} // private constructor
    ~EmotionService() {}
    EmotionService(const EmotionService &) = delete; // disable copying
    EmotionService &operator=(const EmotionService &) = delete; // disable assignment

    std::vector<std::string> vt_label;
    const float mean_vals[3] = {103.94f, 116.78f, 123.68f};
    const float norm_vals[3] = {0.017f,0.017f,0.017f};
    //! @brief NCNN network
    ncnn::Net mobilenet;

public:
    static EmotionService *getInstance();
    
    //! @brief
    bool init_models( std::string path_file_model);
    bool getLabels(std::string file_path);
    std::string predict(const cv::Mat& img);
    

};


