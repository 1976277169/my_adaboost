#include "cascade_classifier.h"


void init_cascade_classifier(CascadeClassifier *cascade, std::vector<StrongClassifier*> scs, int WIDTH, int HEIGHT)
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


int classify(CascadeClassifier *cascade, Sample *sample)
{
    int size = cascade->scs.size();

    for(int i = 0; i < size; i++){
        if(classify(cascade->scs[i], sample) == 0)
            return 0;
    }

    return 1;
}


float fnr(CascadeClassifier *cc, std::vector<Sample*> &samples){
    int size = samples.size();
    int fn = 0;

    assert(size > 0);

    for(int i = 0; i < size; i++){
        if(classify(cc, samples[i]) == 0)
            fn ++;
    }

    return float(fn) / size;
}


float fpr(CascadeClassifier *cc, std::vector<Sample*> &samples){
    int size = samples.size();
    int fp = 0;

    assert(size > 0);

    for(int i = 0; i < size; i++){
        if(classify(cc, samples[i]) == 1)
            fp ++;
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

    for(int i = 0; i < size; i++)
        save(cascade->scs[i], fout);

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


void clear(CascadeClassifier **aCC)
{
    CascadeClassifier *cascade = *aCC;
    int size = cascade->scs.size();

    for(int i = 0; i < size; i++)
        clear(&cascade->scs[i]);

    cascade->scs.clear();

    delete *aCC;
    *aCC = NULL;
}


void clean_samples(CascadeClassifier *cc, std::vector<Sample*> &samples){
    int size = samples.size();
    int scSize = cc->scs.size();

    for(int i = 0; i < size; i++){
        Sample *sample = samples[i];

        for(int j = 0; j < scSize; j++){
            if(classify(cc->scs[j], samples[i]) == 0){
                release_sample(&samples[i]);

                HU_SWAP(samples[i], samples[size - 1], Sample*);
                i--;
                size --;
                break;
            }
        }
    }

    samples.erase(samples.begin() + size, samples.end());
}

