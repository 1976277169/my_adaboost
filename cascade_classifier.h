#ifndef _CASCADE_CLASSIFIER_H_
#define _CASCADE_CLASSIFIER_H_

#include "strong_classifier.h"
#include <list>

typedef struct {
    std::list<StrongClassifier*> scs;
    int WIDTH, HEIGHT;
} CascadeClassifier;

void init_cascade_classifier(CascadeClassifier *cascade, std::list<StrongClassifier*> &scs, int WIDTH, int HEIGHT);

void add(CascadeClassifier *cascade, StrongClassifier *sc);
void del(CascadeClassifier *cascade);

int classify(CascadeClassifier *cascade, float *img, int stride, int x, int y);

float fnr(CascadeClassifier *cascade, std::list<float*> &posSamples, int stride);
float fpr(CascadeClassifier *cascade, std::list<float*> &negSamples, int stride);

void load(CascadeClassifier **aCascade, const char *fileName);
void save(CascadeClassifier *cascade, const char *fileName);
void clear(CascadeClassifier *cascade);

#endif
