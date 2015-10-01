#include "tool.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

#define LT(a, b) (a < b)
#define BG(a, b) (a > b)

IMPLEMENT_QSORT(sort_arr_float_ascend, float, LT)
IMPLEMENT_QSORT(sort_arr_float_descend, float, BG)

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
    float sum = 0; 
    int sampleSize = numPos + numNeg;

    for(int i = 0; i < sampleSize; i++)
        sum += weights[i];

    for(int i = 0; i < sampleSize; i++)
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


void print_feature_list(std::vector<Feature *> &featureSet, const char *fileName)
{
    long size = featureSet.size();
    std::vector<Feature*>::iterator iter = featureSet.begin();

    FILE *fout = fopen(fileName, "w");
    assert(fout != NULL);

    for(long i = 0; i < size; i++, iter++)
    {
        Feature *feat = *iter;

        fprintf(fout, "%d %2d %2d %2d %2d\n", feat->type, feat->x0, feat->y0, feat->w, feat->h);
    }

    fclose(fout);
}


void show_image(float *data, int width, int height)
{
    cv::Mat img(height, width, CV_32FC1, data);

    cv::imshow("img", img);
    cv::waitKey();
}
