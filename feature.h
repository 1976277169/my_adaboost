#ifndef _FEATURE_H_
#define _FEATURE_H_

#include <vector>
#include <list>
#include <stdio.h>

#define SAMPLE_FEATURE_MEMEORY_SIZE 2 //base memory 1G

typedef enum {VERTICAL_2 = 0, HORIZONTAL_2, VERTICAL_3, HORIZONTAL_3, CROSS} HaarFeatureType;

typedef struct
{
    int type;
    int x0, y0;
    int w, h;
} Feature;

void init_feature(Feature *f, int type, int x, int y, int width, int height);
float get_value(Feature *f, float *img, int stride, int x, int y);
void generate_feature_set(std::vector<Feature*> &featSet, const int WIDTH, const int HEIGHT);

void extract_sample_features(const char *outfile, std::vector<Feature*> &featSet,
            std::list<float*> &positiveSet,
            std::list<float *> &negativeSet, int stride);

long init_feature_buffer_size(long featureSetSize, int sampleSize);
#endif

