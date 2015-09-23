#ifndef _FEATURE_H_
#define _FEATURE_H_

#include <vector>
#include <list>
#include <stdio.h>
#include <assert.h>

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


#endif

