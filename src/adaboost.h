#ifndef _ADA_BOOST_H_
#define _ADA_BOOST_H_

#include "classifier.h"


typedef struct {
    std::vector<std::string> bufList;
    std::vector<std::string> imgList;

    cv::Mat *imgs;
    int bufSize;

    int id;
    int scale;
    int level;
    int tflag;
} NegSetGenerator;


typedef struct{
    StrongClassifier *sc;
    int WINW, WINH;
    int ssize;

    float startScale;
    float endScale;
    int layer;
    float offsetFactor;
}Cascade;


typedef struct {
    int x, y;
    int width;
    int height;
} HRect;

int train(Cascade *cc, const char *posFilePath, const char *negFilePath);
int predict(Cascade *cc, uint32_t *iImg, int iStride, int winSize, float &score);

void init_detect_factor(Cascade *cc, float startScale, float endScale, float offset, int layer);
int detect(Cascade *cc, uint8_t *img, int width, int height, int stride, HRect **resRect);

int save(Cascade *cc, const char *filePath);
int load(Cascade *cascade, const char *filePath);

void release_data(Cascade *cc);
void release(Cascade **cc);

void release_data(NegSetGenerator *ng);
void release(NegSetGenerator **ng);

#endif
