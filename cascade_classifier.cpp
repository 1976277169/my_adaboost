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

    if(cascade->scs.size() == 0)
        return 0;

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
        if(classify(cascade, *iter, stride, 0, 0) == 1)
            fp ++;

        iter++;
    }

    return float(fp) / size;
}


void save(CascadeClassifier *cascade, const char *fileName)
{
    FILE *fout = fopen(fileName, "wb");

    if(fout == NULL)
    {
        printf("Can't write file %s\n", fileName);
        return ;
    }

    int size = cascade->scs.size();

    fwrite(&size, sizeof(int), 1, fout);
    fwrite(&cascade->WIDTH, sizeof(int), 1, fout);
    fwrite(&cascade->HEIGHT, sizeof(int), 1, fout);

    std::list<StrongClassifier*>::iterator iter = cascade->scs.begin();

    for(int i = 0; i < size; i++, iter++)
        save(*iter, fout);

    fclose(fout);
}


void load(CascadeClassifier **aCascade, const char *fileName)
{
    FILE *fin = fopen(fileName, "rb");
    if(fin == NULL)
    {
        printf("Can't read file %s\n", fileName);
        return ;
    }

    int size;

    int ret = fread(&size, sizeof(int), 1, fin);

    CascadeClassifier *cascade = new CascadeClassifier;

    ret = fread(&cascade->WIDTH, sizeof(int), 1, fin);
    ret = fread(&cascade->HEIGHT, sizeof(int), 1, fin);

    for(int i = 0; i < size; i++)
    {
        StrongClassifier *sc;
        load(&sc, fin);

        cascade->scs.push_back(sc);
    }

    fclose(fin);

    *aCascade = cascade;
}


void clear(CascadeClassifier *cascade)
{
    std::list<StrongClassifier*>::iterator iter = cascade->scs.begin();
    std::list<StrongClassifier*>::iterator iterEnd = cascade->scs.end();

    while(iter != iterEnd)
    {
        clear(*iter);
        iter++;
    }

    cascade->scs.clear();

    delete cascade;
}
