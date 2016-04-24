#include "sample.h"
#include <stdio.h>

Sample *create_sample(uint8_t *img, int width, int height, int stride){
    uint8_t *ptrImg8U = NULL;
    float *ptr = NULL;
    Sample *sample = new Sample;

    memset(sample, 0, sizeof(Sample));

    sample->width = width;
    sample->height = height;
    sample->stride = width;
    sample->stride2 = stride + 1;

    sample->img = new uint8_t[width * height];
    sample->intImg = new float[sample->stride2 * (height + 1)];

    sample->score = 0;

    if(img == NULL)
        return sample;

    ptrImg8U = sample->img;

    for(int y = 0; y < height; y++){
        memcpy(ptrImg8U, img, sizeof(uint8_t) * width);

        ptrImg8U += width;
        img += stride;
    }

    memset(sample->intImg, 0, sizeof(float) * sample->stride2 * (height + 1));

    ptr = sample->intImg + (sample->stride2 + 1);
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            ptr[x] = ptr[x - 1] + sample->img[y * width + x] / 255.0;
        }

        ptr += sample->stride2;
    }

    ptr = sample->intImg + (sample->stride2 + 1);
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            ptr[x] += ptr[x - sample->stride2];
        }
        ptr += sample->stride2;
    }


    return sample;
}


void set_image(Sample* sample, uint8_t *img, int width, int height, int stride){
    if(sample->width == width && sample->height == height){
        for(int y = 0; y < height; y++){
            memcpy(sample->img + y * sample->stride, img + y * stride, sizeof(uint8_t) * width);
        }

        memset(sample->intImg, 0, sizeof(float) * sample->stride2 * (height + 1));

    }
    else {
        delete [] sample->img;
        sample->stride = ((width + 3) >> 2) << 2;
        sample->img = new uint8_t[sample->stride * height];

        delete [] sample->intImg;
        sample->stride2 = ((width + 1 + 3) >> 2) << 2;
        sample->intImg = new float[sample->stride2 * (height + 1)];

        memset(sample->intImg, 0, sizeof(float) * sample->stride2 * (height + 1));
    }

    float *ptr = sample->intImg + (sample->stride2 + 1);
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            ptr[x] = ptr[x - 1] + sample->img[y * width + x] / 255.0;
        }

        ptr += sample->stride2;
    }

    ptr = sample->intImg + (sample->stride2 + 1);
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            ptr[x] += ptr[x - sample->stride2];
        }
        ptr += sample->stride2;
    }

    sample->score = 0;
}


void release_sample(Sample** src){
    Sample *sample = *src;

    if(sample->img  != NULL){
        delete [] sample->img;
        sample->img = NULL;
    }

    if(sample->intImg != NULL){
        delete [] sample->intImg;
        sample->intImg = NULL;
    }

    delete *src;
    *src = NULL;
}


int read_positive_sample_from_file(const char *filePath, int WINW, int WINH, std::vector<Sample*> &samples){
    std::vector<std::string> imgList;

    int ret = read_image_list(filePath, imgList);

    if(ret != 0) return 1;

    int size = imgList.size();

    for(int i = 0; i < size; i++){
        cv::Mat img = cv::imread(imgList[i], 0);

        if(img.empty()){
            printf("Can't read image %s\n", imgList[i].c_str());
            return 2;
        }

        cv::resize(img, img, cv::Size(WINW, WINH));

        Sample *sample = create_sample(img.data, img.cols, img.rows, img.step);

        assert(sample != NULL);

        samples.push_back(sample);
    }

    return 0;
}


void select_samples(std::vector<Sample*> &samples, float thresh){
    std::vector<Sample*> res;
    int size = samples.size();

    for(int i = 0; i < size; i++){
        if(samples[i]->score > thresh)
            res.push_back(samples[i]);
        else
            release_sample(&samples[i]);
    }

    samples.clear();
    samples = res;
}


void clear_list(std::vector<Sample*> &samples){
    int size = samples.size();

    for(int i = 0; i < size; i++){
        release_sample(&samples[i]);
    }

    samples.clear();
}


void init_feature(Feature *f, int type, int x, int y, int width, int height)
{
    f->type = type;
    f->x0 = x;
    f->y0 = y;

    f->w = width;
    f->h = height;
}


void init_feature(Feature* dstFeat, Feature *srcFeat)
{
    dstFeat->type = srcFeat->type;
    dstFeat->x0 = srcFeat->x0;
    dstFeat->y0 = srcFeat->y0;
    dstFeat->w = srcFeat->w;
    dstFeat->h = srcFeat->h;
}

#define SUM_FIELD(sample, x, y, w, h, value) \
{ \
    int stride2 = sample->stride2;             \
    float *ptr = sample->intImg + stride2 + 1; \
    int ax0 = x - 1;                            \
    int ay0 = y - 1;                            \
    int ax1 = x + w - 1;                        \
    int ay1 = y + h - 1;                        \
    value = ptr[ay1 * stride2 + ax1] - ptr[ay1 * stride2 + ax0] - ptr[ay0 * stride2 + ax1] + ptr[ay0 * stride2 + ax0]; \
}



void extract_feature_values(Feature *f, std::vector<Sample*> &samples, float *values){
    int x0 = f->x0;
    int y0 = f->y0;

    int w = f->w;
    int h = f->h;

    int dx, dy;
    float lef, rig, cen, top, bot;

    int size = samples.size();

    switch(f->type){
        case VERTICAL_2:
            dx = w >> 1;
            for(int i = 0; i < size; i++){
                Sample *sample = samples[i];
                SUM_FIELD(sample, x0,      y0, dx, h, lef);
                SUM_FIELD(sample, (x0 + dx), y0, dx, h, rig);

                values[i] = lef - rig;
            }

            break;

        case HORIZONTAL_2:
            dy = h >> 1;
            for(int i = 0; i < size; i++){
                Sample *sample = samples[i];
                SUM_FIELD(sample, x0, y0,      w, dy, top);
                SUM_FIELD(sample, x0, (y0 + dy), w, dy, bot);

                values[i] = top - bot;
            }

            break;

        case VERTICAL_3:
            dx = w / 3;
            for(int i = 0; i < size; i++){
                Sample *sample = samples[i];
                SUM_FIELD(sample, x0     , y0, dx, h, lef);
                SUM_FIELD(sample, (x0 + dx), y0, dx, h, cen);
                SUM_FIELD(sample, (x0 + 2 * dx), y0, dx, h, rig);

                values[i] = cen - lef - rig;
            }
            break;

        case HORIZONTAL_3:
            dy = h / 3;
            for(int i = 0; i < size; i++){
                Sample *sample = samples[i];
                SUM_FIELD(sample, x0, y0, w, dy, top);
                SUM_FIELD(sample, x0, (y0 + dy), w, dy, cen);
                SUM_FIELD(sample, x0, (y0 + 2 * dy), w, dy, bot);

                values[i] = cen - top - bot;
            }

            break;

        case CROSS:
            dx = w >> 1;
            dy = h >> 1;

            float lt, rt, lb, rb;

            for(int i = 0; i < size; i++){
                Sample *sample = samples[i];
                SUM_FIELD(sample, x0, y0, dx, dy, lt);
                SUM_FIELD(sample, (x0 + dx), y0, dx, dy, rt);
                SUM_FIELD(sample, x0, (y0 + dy), dx, dy, lb);
                SUM_FIELD(sample, (x0 + dx), (y0 + dy), dx, dy, rb);

                values[i] = lt + rb - lb - rt;
            }

            break;
    }
}


float get_value(Sample *sample, Feature *feat){
    int x0 = feat->x0;
    int y0 = feat->y0;

    int w = feat->w;
    int h = feat->h;

    int dx, dy;
    float lef, rig, cen, top, bot;
    float lt, rt, lb, rb;


    switch(feat->type){
        case VERTICAL_2:
            dx = w >> 1;

            SUM_FIELD(sample, x0,      y0, dx, h, lef);
            SUM_FIELD(sample, (x0 + dx), y0, dx, h, rig);

            return lef - rig;

        case HORIZONTAL_2:
            dy = h >> 1;

            SUM_FIELD(sample, x0, y0,      w, dy, top);
            SUM_FIELD(sample, x0, (y0 + dy), w, dy, bot);

            return top - bot;

        case VERTICAL_3:
            dx = w / 3;

            SUM_FIELD(sample, x0     , y0, dx, h, lef);
            SUM_FIELD(sample, (x0 + dx), y0, dx, h, cen);
            SUM_FIELD(sample, (x0 + 2 * dx), y0, dx, h, rig);

            return cen - lef - rig;

        case HORIZONTAL_3:
            dy = h / 3;

            SUM_FIELD(sample, x0, y0, w, dy, top);
            SUM_FIELD(sample, x0, (y0 + dy), w, dy, cen);
            SUM_FIELD(sample, x0, (y0 + 2 * dy), w, dy, bot);

            return cen - top - bot;

        case CROSS:
            dx = w >> 1;
            dy = h >> 1;

            SUM_FIELD(sample, x0, y0, dx, dy, lt);
            SUM_FIELD(sample, (x0 + dx), y0, dx, dy, rt);
            SUM_FIELD(sample, x0, (y0 + dy), dx, dy, lb);
            SUM_FIELD(sample, (x0 + dx), (y0 + dy), dx, dy, rb);

            return lt + rb - lb - rt;
    }
}


void write_samples(std::vector<Sample*> &samples, const char *outdir){
    char outfile[256], command[128];
    int ret = 0, size;

    sprintf(command, "mkdir -p %s", outdir);
    ret = system(command);

    sprintf(command, "rm -f %s/*", outdir);
    ret = system(command);

    size = samples.size();

    for(int i = 0; i < size; i++){
        Sample *sample = samples[i];
        cv::Mat img(sample->height, sample->width, CV_8UC1, sample->img);

        sprintf(outfile, "%s/%d.jpg", outdir, i);
        cv::imwrite(outfile, img);
    }
}


/****************************************************
 * generate vertical 2 feature
 * width step 2
 * height step 1
 ****************************************************/
static void generate_feature_set_type_vertical_2(std::vector<Feature*> &featSet, const int WIDTH, const int HEIGHT)
{
    int minWidth = 4;
    int height = 4;
    int width = 4;

    while(height <= HEIGHT)
    {
        width = minWidth;
        while(width <= WIDTH)
        {
            int y = 0;

            while(y + height <= HEIGHT)
            {
                int x = 0;

                while(x + width <= WIDTH)
                {
                    Feature *f = new Feature;
                    init_feature(f, VERTICAL_2, x, y, width, height);
                    featSet.push_back(f);

                    x++;
                }

                y++;
            }

            width += 2;
        }

        height ++;
    }
}


/****************************************************
 * generate horizontal 2 feature
 * width step 1
 * height step 2
 ****************************************************/
static void generate_feature_set_type_horizontal_2(std::vector<Feature*> &featSet, const int WIDTH, const int HEIGHT)
{
    int minWidth = 4;
    int height = 4;
    int width = 4;

    while(height <= HEIGHT)
    {
        width = minWidth;
        while(width <= WIDTH)
        {
            int y = 0;

            while(y + height <= HEIGHT)
            {
                int x = 0;

                while(x + width <= WIDTH)
                {
                    Feature *f = new Feature;
                    init_feature(f, HORIZONTAL_2, x, y, width, height);
                    featSet.push_back(f);

                    x++;
                }

                y++;
            }

            width ++;
        }

        height += 2;
    }
}


/****************************************************
 * generate vertical 3 feature
 * width step 3
 * height step 1
 ****************************************************/
static void generate_feature_set_type_vertical_3(std::vector<Feature*> &featSet, const int WIDTH, const int HEIGHT)
{
    int minWidth = 3;
    int height = 4;
    int width = 3;

    while(height <= HEIGHT)
    {
        width = minWidth;
        while(width <= WIDTH)
        {
            int y = 0;

            while(y + height <= HEIGHT)
            {
                int x = 0;

                while(x + width <= WIDTH)
                {
                    Feature *f = new Feature;
                    init_feature(f, VERTICAL_3, x, y, width, height);
                    featSet.push_back(f);

                    x++;
                }

                y++;
            }

            width += 3;
        }

        height ++;
    }
}


/****************************************************
 * generate horizontal 3 feature
 * width step 1
 * height step 3
 ****************************************************/
static void generate_feature_set_type_horizontal_3(std::vector<Feature*> &featSet, const int WIDTH, const int HEIGHT)
{
    int minWidth = 4;
    int height = 3;
    int width = 4;

    while(height <= HEIGHT)
    {
        width = minWidth;
        while(width <= WIDTH)
        {
            int y = 0;

            while(y + height <= HEIGHT)
            {
                int x = 0;

                while(x + width <= WIDTH)
                {
                    Feature *f = new Feature;
                    init_feature(f, HORIZONTAL_3, x, y, width, height);
                    featSet.push_back(f);

                    x++;
                }

                y++;
            }

            width ++;
        }

        height += 3;
    }
}


/****************************************************
 * generate cross feature
 * width step 4
 * height step 4
 ****************************************************/
static void generate_feature_set_cross(std::vector<Feature*> &featSet, const int WIDTH, const int HEIGHT)
{
    int minWidth = 4;
    int height = 4;
    int width = 4;

    while(height <= HEIGHT)
    {
        width = minWidth;
        while(width <= WIDTH)
        {
            int y = 0;

            while(y + height <= HEIGHT)
            {
                int x = 0;

                while(x + width <= WIDTH)
                {
                    Feature *f = new Feature;
                    init_feature(f, CROSS, x, y, width, height);
                    featSet.push_back(f);

                    x++;
                }

                y++;
            }

            width += 4;
        }

        height += 4;
    }
}


void generate_feature_set(std::vector<Feature*> &featSet, const int WIDTH, const int HEIGHT)
{
    generate_feature_set_type_vertical_2(featSet, WIDTH, HEIGHT);
    generate_feature_set_type_horizontal_2(featSet, WIDTH, HEIGHT);
    generate_feature_set_type_vertical_3(featSet, WIDTH, HEIGHT);
    generate_feature_set_type_horizontal_3(featSet, WIDTH, HEIGHT);

    generate_feature_set_cross(featSet, WIDTH, HEIGHT);
}


void clear_features(std::vector<Feature*> &featSet)
{
    int size = featSet.size();

    for(int i = 0; i < size; i++)
        delete featSet[i];

    featSet.clear();
}


void print_feature_list(std::vector<Feature *> &featureSet, FILE *fout)
{
    long size = featureSet.size();
    std::vector<Feature*>::iterator iter = featureSet.begin();

    assert(fout != NULL);

    for(long i = 0; i < size; i++, iter++)
    {
        Feature *feat = *iter;

        fprintf(fout, "%d %2d %2d %2d %2d\n", feat->type, feat->x0, feat->y0, feat->w, feat->h);
    }
    fflush(stdout);
}


void print_feature(FILE *fout, Feature *feat){
    fprintf(fout, "type: %d\n", feat->type);
    fprintf(fout, "x0 = %2d, y0 = %2d, w = %2d, h = %2d\n", feat->x0, feat->y0, feat->w, feat->h);
    fflush(fout);
}
