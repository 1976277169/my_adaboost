#ifndef _SAMPLE_H_
#define _SAMPLE_H_

#include "tool.h"


typedef enum {VERTICAL_2 = 0, HORIZONTAL_2, VERTICAL_3, HORIZONTAL_3, CROSS} HaarFeatureType;

typedef struct
{
    int type;
    int x0, y0;
    int w, h;
} Feature;


typedef struct {
    uint8_t *img;
    float *intImg; //intgral image;

    int width;
    int height;
    int stride;
    int stride2;

    float score;
} Sample;


Sample* create_sample(uint8_t *img, int width, int height, int stride);
void set_image(Sample* sample, uint8_t *img, int width, int height, int stride);
void release_sample(Sample** src);
int read_positive_sample_from_file(const char *filePath, int WINW, int WINH, std::vector<Sample*> &samples);
void select_samples(std::vector<Sample*> &samples, float thresh);
void clear_list(std::vector<Sample*> &samples);
void write_samples(std::vector<Sample*> &samples, const char *outdir);

void init_feature(Feature *f, int type, int x, int y, int width, int height);
void init_feature(Feature* dstFeat, Feature *srcFeat);
void clear_features(std::vector<Feature*> &featSet);

void generate_feature_set(std::vector<Feature*> &featSet, const int WIDTH, const int HEIGHT);

float get_value(Sample *sample, Feature *feat);
void extract_feature_values(Feature *f, std::vector<Sample*> &samples, float *values);

void print_feature(FILE *fout, Feature *feat);
void print_feature_list(std::vector<Feature *> &featureSet, FILE *fout);


#endif
