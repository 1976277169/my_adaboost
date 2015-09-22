#include "cascade_classifier.h"


void init_cascade_classifier(CascadeClassifier *cascade, std::list<StrongClassifier*> &scs, int WIDTH, int HEIGHT)
{
    cascade->scs = scs;
    cascade->HEIGHT = HEIGHT;
    cascade->WIDTH = WIDTH;
}


void add(CascadeClassifier *cascade, StrongClassifier *sc)
{
    cascade->scs.push_back(sc);
}


void del(CascadeClassifier *cascade)
{
    cascade->scs.pop_back();
}


int classify(CascadeClassifier *cascade, float *img, int stride, int x, int y)
{
    std::list<StrongClassifier*>::iterator iter = cascade->scs.begin();
    std::list<StrongClassifier*>::iterator iterEnd = cascade->scs.end();

    if(cascade->scs.size() == 0) return 0;

    while(iter != iterEnd){
        if(classify(*iter, img, stride, x, y) == 0)
            return 0;
        iter++;
    }

    return 1;
}


float fnr(CascadeClassifier *cascade, std::list<float*> &posSamples, int stride)
{
    int size = posSamples.size();
    int fn = 0;

    std::list<float*>::iterator iter = posSamples.begin();
    std::list<float*>::iterator iterEnd = posSamples.end();

    while(iter != iterEnd)
    {
        if(classify(cascade, *iter, stride, 0, 0) == 0)
            fn ++;

        iter++;
    }

    return float(fn) / size;
}


float fpr(CascadeClassifier *cascade, std::list<float*> &negSamples, int stride)
{
    int size = negSamples.size();
    int fp = 0;

    std::list<float*>::iterator iter = negSamples.begin();
    std::list<float*>::iterator iterEnd = negSamples.end();

    while(iter != iterEnd)
    {
        if(classify(cascade, *iter, stride, 0, 0) == 0)
            fp ++;

        iter++;
    }

    return float(fp) / size;
}
