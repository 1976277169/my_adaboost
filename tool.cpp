#include "tool.h"
#include <math.h>
#include <assert.h>

#define LT(a, b) ((a) < (b))
#define BG(a, b) ((a) > (b))
#define LT_PAIR_V(a, b) ((a).value < (b).value)
#define LT_PAIR_I(a, b) ((a).idx < (b).idx)

IMPLEMENT_QSORT(sort_arr_float_ascend, float, LT)
IMPLEMENT_QSORT(sort_arr_float_descend, float, BG)
IMPLEMENT_QSORT(sort_arr_pair, PairF, LT_PAIR_V)
IMPLEMENT_QSORT(sort_arr_pair_idx, PairF, LT_PAIR_I)

int read_image_list(const char *fileName, std::vector<std::string> &imageList)
{
    char line[512];
    FILE *fp = fopen(fileName, "r");

    if(fp == NULL)
    {
        printf("Can't read file %s.\n", fileName);
        return 1;
    }

    while(fscanf(fp, "%s\n", line) != EOF)
    {
        imageList.push_back(std::string(line));
    }

    return 0;
}


void normalize_image(float *img, int width, int height)
{
    int sq = width * height;

    double mean = 0, stand = 0;

    for(int i = 0; i < sq; i++)
        mean += img[i];
    mean /= sq;

    for(int i = 0; i < sq; i++)
        stand += (mean - img[i]) * (mean - img[i]);

    stand = sqrt(stand / sq);

    for(int i = 0; i < sq; i++)
        img[i] = (img[i] - mean) / stand;
}


void normalize_image_npd(float *img, int width, int height)
{
    float mean = 0;
    int size = width * height;

    for(int i = 0; i < size; i++)
        mean += img[i];

    mean /= size;

    for(int i = 0; i < size; i++)
        img[i] = (mean - img[i]) / (mean + img[i] + 0.00001);
}


float* rotate_90deg(float *img, int width, int height)
{
    float *img90deg = new float[width*height];
    float *ptrImg = img;

    for (int y = 0; y < height; y++)
    {
        int idx = height - 1 - y;

        for (int x = 0; x < width; x++)
            img90deg[idx + x * height] = ptrImg[x];
        ptrImg += width;
    }

    return img90deg;
}


float* rotate_180deg(float *img, int width, int height)
{
    float *img180deg = new float[width*height];
    float *ptrImg = img;

    for (int y = 0; y < height; y++)
    {
        int idx = (height - 1 - y) * width;

        for (int x = 0; x < width; x++)
            img180deg[idx + width - 1 - x] = ptrImg[x];

        ptrImg += width;
    }

    return img180deg;
}



float* rotate_270deg(float *img, int width, int height)
{
    float *img270deg = new float[width*height];
    float *ptrImg = img;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
            img270deg[y+((width-x-1)*height)] = ptrImg[x];

        ptrImg += width;
    }

    return img270deg;
}


float* vertical_mirror(float *img, int width, int height)
{
    float *vmirror = new float[width * height];
    float *ptrImg = img;

    for(int y = 0; y < height; y++)
    {
        int idx = y * width;

        for(int x = 0; x < width; x++)
            vmirror[idx + width - 1 - x] = ptrImg[x];
        ptrImg += width;
    }

    return vmirror;
}


void add_rotated_images(std::list<float*> &set, int size, int width, int height)
{
    std::list<float*>::iterator it = set.begin();
    for (int i = 0; i < size; i++, it++)
    {
        set.push_back(rotate_90deg(*it, width, height));
        set.push_back(rotate_180deg(*it, width, height));
        set.push_back(rotate_270deg(*it, width, height));
    }
}


void add_vertical_mirror(std::list<float*> &set, int size, int width, int height)
{
    float *vmirror;
    int i, j, x, y;

    std::list<float*>::iterator it = set.begin();
    for (i = 0; i < size; i++, it++)
        set.push_back(vertical_mirror(*it, width, height));
}


void integral_image(float *img, int width, int height)
{
    float *ptrImg = img;
    for(int y = 0; y < height; y++)
    {
        for(int x = 1; x < width; x++)
            ptrImg[x] += ptrImg[x - 1];
        ptrImg += width;
    }

    for(int x = 0; x < width; x++)
    {
        ptrImg = img + x;
        for(int y = 1; y < height; y++)
            ptrImg[y * width] += ptrImg[(y - 1) * width];
    }
}


void init_steps_false_positive(float **Fi, int step, float targetFPR)
{
    float *arr = NULL;
    assert(step > 0);

    arr = new float[step];

    if(step > 2)
    {
        float fprPerStep = pow(targetFPR / 0.125, 1 / float(step - 2));

        arr[0] = 0.5;
        arr[1] = 0.25;

        for(int i = 2; i < step; i++)
            arr[i] = fprPerStep;
    }
    else if(step == 2)
    {
        arr[0] = 0.5;
        arr[1] = targetFPR;
    }
    else
        arr[0] = targetFPR;

    *Fi = arr;
}


void init_weights(float **weights, int numPos, int numNeg)
{
    float t = 0;
    int sampleSize = numPos + numNeg;

    *weights = new float[sampleSize];

    t = 0.5 / numPos;
    for(int i = 0; i < numPos; i++)
        (*weights)[i] = t;

    t = 0.5 / numNeg;
    for(int i = numPos; i < sampleSize; i++)
        (*weights)[i] = t;
}


void update_weights(float *weights, int numPos, int numNeg)
{
    double sum = 0;
    int sampleSize = numPos + numNeg;

    sum = 0;
    for(int i = 0; i < numPos; i++)
        sum += weights[i];

    sum *= 2;
    for(int i = 0; i < numPos; i++)
        weights[i] /= sum;

    sum = 0;
    for(int i = numPos; i < sampleSize; i++)
        sum += weights[i];

    sum *= 2;
    for(int i = numPos; i < sampleSize; i++)
        weights[i] /= sum;
}


void clear_list(std::list<float*> &set)
{
    std::list<float*>::iterator iter = set.begin();
    std::list<float*>::iterator iterEnd = set.end();

    while(iter != iterEnd)
    {
        if((*iter) != NULL)
            delete[] (*iter);
        iter++;
    }

    set.clear();
}


void show_image(float *data, int width, int height)
{
    cv::Mat img(height, width, CV_32FC1, data);

    cv::imshow("img", img);
    cv::waitKey();
}


void print_time(clock_t t)
{
    long sec, min, hour;

    sec = t / CLOCKS_PER_SEC;

    min = sec / 60;
    sec %= 60;

    hour = min / 60;
    min %= 60;

    printf("%02ld:%02ld:%02ld", hour, sec, min);
}


void merge_rect(std::vector<cv::Rect> &rects)
{
    int size = rects.size();
    int *flags = NULL;
    if(size == 0)
        return;
    flags = new int[size];

    memset(flags, 0, sizeof(int) * size);

    for(int i = 0; i < size; i++)
    {
        int xi0 = rects[i].x;
        int yi0 = rects[i].y;
        int xi1 = rects[i].x + rects[i].width - 1;
        int yi1 = rects[i].y + rects[i].height - 1;

        int cix = (xi0 + xi1) / 2;
        int ciy = (yi0 + yi1) / 2;
        int sqi = rects[i].width * rects[i].height;

        for(int j = i + 1; j < size; j++)
        {
            int xj0 = rects[j].x;
            int yj0 = rects[j].y;
            int xj1 = rects[j].x + rects[j].width - 1;
            int yj1 = rects[j].y + rects[j].height - 1;

            int cjx = (xj0 + xj1) / 2;
            int cjy = (yj0 + yj1) / 2;

            int sqj = rects[j].width * rects[j].height;

            if ( ( (xi0 <= cjx && cjx <= xi1) && (yi0 <= cjy && cjy <= yi1) ) ||
                   ( (xj0 <= cix && cix <= xj1) && (yj0 <= ciy && ciy <= yj1) ) )
            {
                if(sqj > sqi)
                    flags[i] = 1;
                else
                    flags[j] = 1;

            }
        }
    }

    std::vector<cv::Rect> tmp;

    for(int i = 0; i < size; i++)
        if(flags[i] == 0)
            tmp.push_back(rects[i]);

    rects = tmp;

    delete []flags;
}


float *mat_to_float(cv::Mat &img)
{
    int w = img.cols;
    int h = img.rows;

    float *data = new float[w * h];
    float *pdata = data;

    uchar *pImg = img.data;

    for(int y = 0; y < h; y++){
        for(int x = 0; x < w; x++)
            pdata[x] = pImg[x] / 255.0;

        pdata += w;
        pImg += img.step;
    }

    return data;
}


