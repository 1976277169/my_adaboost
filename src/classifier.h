#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include "sample.h"


typedef enum {VERTICAL_2 = 0, HORIZONTAL_2, VERTICAL_3, HORIZONTAL_3, CROSS} HaarType;

typedef struct {
    uint16_t x, y;
    uint16_t w, h;

    HaarType type;
} FeatTemp;


typedef struct {
    FeatTemp *featTemps;

    float *wcss; //weak classfier score
    int *wcts; //weak classifier thresh
    int8_t *signs;

    int ssize;
    int capacity;

    float thresh;
} StrongClassifier;

int generate_feature_templates(int WINW, int WINH, FeatTemp **temps);
int train(StrongClassifier *sc, SampleSet *posSet, SampleSet *negSet, float recall, float precision);
int predict(StrongClassifier *sc, uint32_t *intImg, int istride);

#endif
