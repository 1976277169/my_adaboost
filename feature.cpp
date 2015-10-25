#include "feature.h"
#include <assert.h>

#define BLOCK_SUM(intrImg, stride, x, y, w, h) \
        (intrImg[(y + h - 1) * stride + x + w - 1] - \
         (x - 1 >= 0) * intrImg[(y + h - 1) * stride + x - 1] - \
         (y - 1 >= 0) * intrImg[(y - 1) * stride + x + w - 1] + \
         (x - 1 >= 0 && y - 1 >= 0) * intrImg[(y - 1) * stride + x - 1])

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


float get_value(Feature *f, float *img, int stride, int x, int y)
{
    int x0 = x + f->x0;
    int y0 = y + f->y0;
    int w = f->w;
    int h = f->h;

    int dx, dy;
    float lef, rig, cen, top, bot;

    switch(f->type)
    {
        case VERTICAL_2:
            dx = w >> 1;
            lef = BLOCK_SUM(img, stride, x0, y0, dx, h);
            rig = BLOCK_SUM(img, stride, x0 + dx, y0, dx, h);

//            return (lef - rig) / (lef + rig + 0.000001);
            return lef - rig;

        case HORIZONTAL_2:
            dy = h >> 1;
            top = BLOCK_SUM(img, stride, x0, y0, w, dy);
            bot = BLOCK_SUM(img, stride, x0, y0 + dy, w, dy);

//            return (top - bot) / (top + bot + 0.000001);
            return top - bot;

        case VERTICAL_3:
            dx = w / 3;
            lef = BLOCK_SUM(img, stride, x0, y0, dx, h);
            cen = BLOCK_SUM(img, stride, x0 + dx, y0, dx, h);
            rig = BLOCK_SUM(img, stride, x0 + 2 * dx, y0, dx, h);

//            return (cen - lef - rig) / (cen + lef + rig + 0.000001);
            return cen - lef - rig;

        case HORIZONTAL_3:
            dy = h / 3;
            top = BLOCK_SUM(img, stride, x0, y0, w, dy);
            cen = BLOCK_SUM(img, stride, x0, y0 + dy, w, dy);
            bot = BLOCK_SUM(img, stride, x0, y0 + 2 * dy, w, dy);

//            return (cen - top - bot) / (cen + lef + rig + 0.000001);
            return cen - top - bot;

        case CROSS:
            dx = w >> 1;
            dy = h >> 1;
/*
            {
                 float t1 = BLOCK_SUM(img, stride, x0, y0, dx, dy) +
                    BLOCK_SUM(img, stride, x0 + dx, y0 + dy, dx, dy);
                 float t2 = BLOCK_SUM(img, stride, x0 + dx, y0, dx, dy) +
                    BLOCK_SUM(img, stride, x0, y0 + dy, dx, dy);
                 return (t1 - t2) / (t1 + t2 + 0.000001);
            }
*/
            return (BLOCK_SUM(img, stride, x0, y0, dx, dy) +
                    BLOCK_SUM(img, stride, x0 + dx, y0 + dy, dx, dy) -
                    BLOCK_SUM(img, stride, x0 + dx, y0, dx, dy) -
                    BLOCK_SUM(img, stride, x0, y0 + dy, dx, dy));
        default:
            assert(0);
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


void print_feature_list(std::vector<Feature *> &featureSet, const char *fileName)
{
    long size = featureSet.size();
    std::vector<Feature*>::iterator iter = featureSet.begin();

    FILE *fout = fopen(fileName, "w");
    assert(fout != NULL);

    for(long i = 0; i < size; i++, iter++)
    {
        Feature *feat = *iter;

        fprintf(fout, "%d %2d %2d %2d %2d\n", feat->type, feat->x0, feat->y0, feat->w, feat->h);
    }

    fclose(fout);
}
