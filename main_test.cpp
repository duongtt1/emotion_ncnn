#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <vector>
#include "iostream"
// #include "platform.h"
#include "ncnn/platform.h"
// #include "net.h"
#include "ncnn/net.h"
using namespace cv;
using namespace std;

#if NCNN_VULKAN
// #include "gpu.h"
#include "ncnn/gpu.h"
#endif // NCNN_VULKAN

static int detect_squeezenet(const cv::Mat &bgr, std::vector<float> &cls_scores)
{

    ncnn::Net squeezenet;
    squeezenet.opt.use_vulkan_compute = 1;
    squeezenet.opt.num_threads = 8;
    squeezenet.opt.use_vulkan_compute = false;
    squeezenet.opt.use_int8_inference = true;
    squeezenet.opt.use_packing_layout = true;
    squeezenet.opt.use_shader_pack8 = false;
    squeezenet.opt.use_image_storage = false;

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    int ret = 0;
    ret = squeezenet.load_param("../mobilenetv2.param");
    if (ret != 0)
    {
        return ret;
    }
    ret = squeezenet.load_model("../mobilenetv2.bin");
    if (ret != 0)
    {
        return ret;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);

    // const float mean_vals[3] = {104.f, 117.f, 123.f};
    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float std_vals[3] = {1 / 58.395f, 1 / 57.12f, 1 / 57.375f};

    in.substract_mean_normalize(mean_vals, std_vals);
    // fprintf(stderr, "input shape: %d %d %d %d\n", in.dims, in.h, in.w, in.c);
    double tTime = (double)cv::getTickCount();
    ncnn::Extractor ex = squeezenet.create_extractor();
    //
    ex.input("input0", in);

    ncnn::Mat out;
    ex.extract("output0", out);
    // fprintf(stderr, "output shape: %d %d %d %d\n", out.dims, out.h, out.w, out.c);

    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
        // std::cout<<"scores: "<<cls_scores[j]<<std::endl;
        // std::cout<<"outs: "<<out[j]<<std::endl;
    }
    tTime = 1000 * ((double)cv::getTickCount() - tTime) / cv::getTickFrequency();
    printf("cost time:%.4fms\n", tTime);
    return 0;
}

static int print_topk(const std::vector<float> &cls_scores, std::vector<int> &index_vec, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int>> vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int>>());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        index_vec.push_back(index);
        // fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

static int load_labels(std::string path, std::vector<std::string> &labels)
{
    FILE *fp = fopen(path.c_str(), "r");

    while (!feof(fp))
    {

        char str[1024];
        fgets(str, sizeof(str), fp); //

        if (feof(fp))
        {
            break;
        }
        labels.push_back(str);
    }

    return 0;
}


int main(int argc, char **argv)
{

    // LOAD image squnense
    FILE *fpos = fopen("../pot.txt", "r");
    if (NULL == fpos)
    {
        printf("open pos.txt is failed\n");
        return -1;
    }

    static char strFileP[1024];
    static char filepathP[1024];

    int framenum = 0;

    while (!feof(fpos))
    {

        fgets(strFileP, sizeof(strFileP), fpos);

        sscanf(strFileP, "%s", filepathP);
        Mat Frame = imread(filepathP);

        framenum++;
        cout << "No. = " << framenum << endl;

        cout << "width = " << Frame.cols << " height = " << Frame.rows << endl;

        if (Frame.empty())
        {
            cout << "read video is failed!" << endl;
            break;
        }
        else
        {

            std::vector<std::string> labels;
            load_labels(
                "../synset_words.txt",
                labels);
#if NCNN_VULKAN1
            ncnn::create_gpu_instance();
#endif // NCNN_VULKAN

            std::vector<float> cls_scores;
            //    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();//记录时间
            //            double tTime = (double) cv::getTickCount();
            detect_squeezenet(Frame, cls_scores);

//            tTime = 1000 * ((double) cv::getTickCount() - tTime) / cv::getTickFrequency();
//            printf("cost time:%.4fms\n", tTime);

//
#if NCNN_VULKAN1
            ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN

            // show results
            std::string Labels;
            std::vector<int> index;
            print_topk(cls_scores, index, 2);

            for (int i = 0; i < index.size() - 1; i++)
            {
                Labels = labels[index[i]];
                std::cout << "labels is ==  " << Labels << std::endl;
            }
        }
    }
#endif
    return 0;
}