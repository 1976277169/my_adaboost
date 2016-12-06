#include "classifier.h"


void init_feature_template(FeatTemp *ft, int x, int y, int w, int h, HaarType type){
    ft->x = x;
    ft->y = y;
    ft->w = w;
    ft->h = h;
    ft->type = type;
}


int get_value(FeatTemp *temp, uint32_t *img, int stride){
    uint16_t x = temp->x;
    uint16_t y = temp->y;
    uint16_t w = temp->w;
    uint16_t h = temp->h;

    int x0, y0, x1, y1;

    int lef, rig, cen, top, bot, lt, rt, lb, rb;
    int len;

    switch(temp->type){
        case VERTICAL_2:
            len = w >> 1;

            x0 = x - 1;
            y0 = y - 1;
            x1 = x + len - 1;
            y1 = y + h - 1;

            lef = img[y1 * stride + x1] - img[y1 * stride + x0] - img[y0 * stride + x1] + img[y0 * stride + x0];

            x0 = x1;
            x1 = x0 + len;

            rig = img[y1 * stride + x1] - img[y1 * stride + x0] - img[y0 * stride + x1] + img[y0 * stride + x0];

            return lef - rig;

        case VERTICAL_3:
            len = w / 3;

            x0 = x - 1;
            y0 = y - 1;
            x1 = x + len - 1;
            y1 = y + h - 1;

            lef = img[y1 * stride + x1] - img[y1 * stride + x0] - img[y0 * stride + x1] + img[y0 * stride + x0];

            x0 = x1;
            x1 = x0 + len;

            cen = img[y1 * stride + x1] - img[y1 * stride + x0] - img[y0 * stride + x1] + img[y0 * stride + x0];

            x0 = x1;
            x1 = x0 + len;

            rig = img[y1 * stride + x1] - img[y1 * stride + x0] - img[y0 * stride + x1] + img[y0 * stride + x0];

            return cen - lef - rig;

        case HORIZONTAL_2:
            len = h >> 1;

            x0 = x - 1;
            y0 = y - 1;
            x1 = x + w - 1;
            y1 = y + len - 1;

            top = img[y1 * stride + x1] - img[y1 * stride + x0] - img[y0 * stride + x1] + img[y0 * stride + x0];

            y0 = y1;
            y1 = y0 + len;

            bot = img[y1 * stride + x1] - img[y1 * stride + x0] - img[y0 * stride + x1] + img[y0 * stride + x0];

            return top - bot;

        case HORIZONTAL_3:
            len = h / 3;

            x0 = x - 1;
            y0 = y - 1;
            x1 = x + w - 1;
            y1 = y + len - 1;

            top = img[y1 * stride + x1] - img[y1 * stride + x0] - img[y0 * stride + x1] + img[y0 * stride + x0];

            y0 = y1;
            y1 = y0 + len;

            cen = img[y1 * stride + x1] - img[y1 * stride + x0] - img[y0 * stride + x1] + img[y0 * stride + x0];

            y0 = y1;
            y1 = y0 + len;

            bot = img[y1 * stride + x1] - img[y1 * stride + x0] - img[y0 * stride + x1] + img[y0 * stride + x0];

            return cen - top - bot;

        case CROSS:
            x0 = x - 1;
            y0 = y - 1;
            x1 = x0 + (w >> 1);
            y1 = y0 + (h >> 1);
            lt = img[y1 * stride + x1] - img[y1 * stride + x0] - img[y0 * stride + x1] + img[y0 * stride + x0];

            x0 = x1;
            x1 = x0 + (w >> 1);
            rt = img[y1 * stride + x1] - img[y1 * stride + x0] - img[y0 * stride + x1] + img[y0 * stride + x0];

            x0 = x - 1;
            y0 = y1;
            x1 = x0 + (w >> 1);
            y1 = y0 + (h >> 1);
            lb = img[y1 * stride + x1] - img[y1 * stride + x0] - img[y0 * stride + x1] + img[y0 * stride + x0];

            x0 = x1;
            x1 = x0 + (w >> 1);
            rb = img[y1 * stride + x1] - img[y1 * stride + x0] - img[y0 * stride + x1] + img[y0 * stride + x0];

            return lt + rb - rt - lb;
    }
}


int generate_feature_templates(int WINW, int WINH, FeatTemp **temps);


#define FEATURE_NUM 2000

void extract_features(SampleSet *set, FeatTemp *featTemps, int fsize, int *feats){
    int ssize = set->ssize;

    for(int i = 0; i < ssize; i++){
        uint32_t *iImg = set->samples[i]->iImg;
        int istride =  set->samples[i]->istride;

        int *ptrFeats = feats + i;

        for(int f = 0; f < fsize; f++){
            ptrFeats[0] = get_value(featTemps + f, iImg, istride);
            ptrFeats += ssize;
        }
    }
}

#define LENGTH 512

void binary_classify_error(int *posFeats, double *pws, int posSize,
        int *negFeats, double *nws, int negSize,
        int &thresh, int8_t &sign, double minError)
{
    double ptable[LENGTH], ntable[LENGTH];
    float minv = INT_MAX, maxv = -INT_MAX, step;

    for(int i = 0; i < posSize; i++){
        minv = HU_MIN(minv, posFeats[i]);
        maxv = HU_MAX(maxv, posFeats[i]);
    }

    for(int i = 0; i < negSize; i++){
        minv = HU_MIN(minv, negFeats[i]);
        maxv = HU_MAX(maxv, negFeats[i]);
    }

    step = (maxv - minv) / (LENGTH - 1);

    for(int i = 0; i < posSize; i++){
        int id = (posFeats[i] - minv) / step;
        ptable[i] += pws[i];
    }

    for(int i = 0; i < negSize; i++){
        int id = (negFeats[i] - maxv) / step;
        ntable[i] + nws[i];
    }

    double nw = 0, pw = 0;
    double lw = 0, rw = 0;
    double error;

    minError = FLT_MAX;
    for(int i = 0; i < LENGTH; i++){
        pw += ptable[i];
        nw += ptable[i];

        lw = (pw + 1 - nw) / 2.0;
        rw = 1.0 - lw;

        error = -lw * log(lw) - rw * log(rw);
        if(error < minError){
            minError = error;
            thresh = minv + i * step;
            sign = (pw < 0.5);
        }
    }
}


void expand_space(StrongClassifier *sc){
    int len = 20;
    if(sc->ssize == sc->capacity){
        sc->capacity = sc->ssize + len;
        FeatTemp *fbuf = new FeatTemp[sc->capacity];
        memcpy(fbuf, sc->featTemps, sizeof(FeatTemp) * sc->ssize);

        delete [] sc->featTemps;
        sc->featTemps = fbuf;

        float *sbuf = new float[sc->capacity * 2];
        memcpy(sbuf, sc->wcss, sizeof(float) * sc->ssize * 2);
        delete [] sc->wcss;
        sc->wcss = sbuf;

        int *tbuf = new int[sc->capacity];
        memcpy(tbuf, sc->wcts, sizeof(int) * sc->ssize);
        delete [] sc->wcts;
        sc->wcts = tbuf;

        int8_t *signs = new int8_t[sc->capacity];
        memcpy(signs, sc->signs, sizeof(int8_t) * sc->ssize);
    }
}


void split(int *posFeats, double *pws, float *pss, int posSize,
        int *negFeats, double *nws, float *nss, int negSize,
        int bestThresh, int bestSign, float score, float beta){
    if(bestSign == 0){
        for(int i = 0; i < posSize; i++){
            if(posFeats[i] > bestThresh){
                pws[i] *= beta;
                pss[i] += score;
            }
            else {
                pss[i] -= score;
            }
        }

        for(int i = 0; i < negSize; i++){
            if(negFeats[i] < bestThresh){
                nws[i] *= beta;
                nss[i] -= score;
            }
            else {
                nss[i] += score;
            }
        }
    }
    else {
        for(int i = 0; i < posSize; i++){
            if(posFeats[i] < bestThresh){
                pws[i] *= beta;
                pss[i] += score;
            }
            else {
                pss[i] -= score;
            }
        }

        for(int i = 0; i < negSize; i++){
            if(negFeats[i] > bestThresh){
                nws[i] *= beta;
                nss[i] -= score;
            }
            else {
                nss[i] += score;
            }
        }
    }
}


int train(StrongClassifier *sc, SampleSet *posSet, SampleSet *negSet, SampleSet *valSet, float recall, float precision){
    float rate = 0.0f;

    int WINW = posSet->winw;
    int WINH = posSet->winh;

    FeatTemp *featTemps;
    FeatTemp *featTemps2;
    int fsize;

    fsize = generate_feature_templates(WINW, WINH, &featTemps);

    featTemps = new FeatTemp[FEATURE_NUM];

    assert(fsize > 0);

    int posSize = posSet->ssize;
    int negSize = negSet->ssize;
    int valSize = valSet->ssize;

    int *posFeats = new int[posSize];
    int *negFeats = new int[negSize];
    int *valFeats = new int[valSize];

    double *pws = new double[posSize];
    double *nws = new double[negSize];

    float *pss = new float[posSize];
    float *nss = new float[negSize];

    cv::RNG rng(cv::getTickCount());

    for(int i = 0; i < posSize; i++){
        pws[i] = 1.0;
        pss[i] = 0.0f;
    }

    for(int i = 0; i < negSize; i++){
        nws[i] = 1.0;
        nss[i] = 0.0f;
    }

    while(rate < precision){
        double minError;
        int8_t bestSign;
        int bestThresh;
        int bestID;


        for(int i = 0; i < FEATURE_NUM; i++){
            int id = rng.uniform(0, fsize - i);
            featTemps2[i] = featTemps[id];

            HU_SWAP(featTemps[fsize - i], featTemps[i], FeatTemp);
        }

        extract_features(posSet, featTemps2, FEATURE_NUM, posFeats);
        extract_features(negSet, featTemps2, FEATURE_NUM, negFeats);

        update_weights(pws, posSize);
        update_weights(nws, negSize);

        for(int i = 0; i < FEATURE_NUM; i++){
            double error;
            int thresh;
            int8_t sign;

            binary_classify_error(posFeats + i * posSize, pws, posSize, negFeats + i * negSize, nws, negSize, thresh, sign, error);

            if(error < minError){
                minError = error;

                bestSign = sign;
                bestThresh = thresh;
                bestID = i;
            }
        }

        double beta = minError / (1 - minError);
        double score = log(1 / beta);

        int *bestPosFeats = posFeats + bestID * posSize;
        int *bestNegFeats = negFeats + bestID * negSize;

        FeatTemp bestTemp = featTemps2[bestID];

        split(posFeats, pws, pss, posSize, negFeats, nws, nss, negSize, bestThresh, bestSign, score, beta);

    }

    delete [] featTemps;
    delete [] featTemps2;
}


int predict(StrongClassifier *sc, uint32_t *intImg, int istride){
    float score = 0;

    for(int i = 0; i < sc->ssize; i++){
        int value = get_value(sc->featTemps + i, intImg, istride);
        score += sc->wcss[(i << 1) + (value > sc->wcts[i])];
    }

    return (score > sc->thresh);
}


//VERTICAL_2
static int generate_feature_templates_type_vertical_2(FeatTemp *temps, int WINW, int WINH){
    int minWidth, height, width;
    int count;

    minWidth = 4;
    height = 4;
    width = 4;

    if(temps == NULL){
        count = 0;
        while(height <= WINH){
            width = minWidth;
            while(width <= WINW){
                int y = 0;

                while(y + height <= WINH){
                    int x = 0;

                    while(x + width <= WINW){
                        count++;
                        x++;
                    }

                    y++;
                }

                width += 2;
            }

            height ++;
        }

        return count;
    }

    count = 0;
    while(height <= WINH){
        width = minWidth;
        while(width <= WINW){
            int y = 0;

            while(y + height <= WINH){
                int x = 0;

                while(x + width <= WINW){
                    init_feature_template(temps + count, x, y, width, height, VERTICAL_2);
                    count++;
                    x++;
                }

                y++;
            }

            width += 2;
        }

        height ++;
    }

    return count;
}


//HORIZONTAL_2
static int generate_feature_templates_type_horizontal_2(FeatTemp *temps, int WINW, int WINH){
    int minWidth, height, width;
    int count;

    minWidth = 4;
    height = 4;
    width = 4;


    if(temps == NULL){
        count = 0;
        while(height <= WINH){
            width = minWidth;
            while(width <= WINW){
                int y = 0;

                while(y + height <= WINH){
                    int x = 0;

                    while(x + width <= WINW){
                        count++;
                        x++;
                    }

                    y++;
                }

                width ++;
            }

            height += 2;
        }

        return count;
    }

    count = 0;

    while(height <= WINH){
        width = minWidth;
        while(width <= WINW){
            int y = 0;

            while(y + height <= WINH){
                int x = 0;

                while(x + width <= WINW){
                    init_feature_template(temps + count, x, y, width, height, HORIZONTAL_2);
                    count++;
                    x++;
                }

                y++;
            }

            width ++;
        }

        height += 2;
    }

    return count;
}


//VERTICAL_3
static int generate_feature_templates_type_vertical_3(FeatTemp *temps, int WIDTH, int HEIGHT){
    int minWidth, height, width;
    int count;

    minWidth = 3;
    height = 4;
    width = 3;

    if(temps == NULL){
        count = 0;

        while(height <= HEIGHT){
            width = minWidth;

            while(width <= WIDTH){
                int y = 0;

                while(y + height <= HEIGHT){
                    int x = 0;

                    while(x + width <= WIDTH){
                        count++;
                        x++;
                    }

                    y++;
                }

                width += 3;
            }

            height ++;
        }
    }

    count = 0;

    while(height <= HEIGHT){
        width = minWidth;

        while(width <= WIDTH){
            int y = 0;

            while(y + height <= HEIGHT){
                int x = 0;

                while(x + width <= WIDTH){
                    init_feature_template(temps + count, x, y, width, height, VERTICAL_3);
                    count++;
                    x++;
                }

                y++;
            }

            width += 3;
        }

        height ++;
    }

    return count;
}


//HORIZONTAL_3
static int generate_feature_templates_type_horizontal_3(FeatTemp *temps, int WIDTH, int HEIGHT){
    int minWidth, height, width;
    int count;

    minWidth = 4;
    height = 3;
    width = 4;

    if(temps == NULL){
        count = 0;

        while(height <= HEIGHT){
            width = minWidth;
            while(width <= WIDTH){
                int y = 0;

                while(y + height <= HEIGHT){
                    int x = 0;

                    while(x + width <= WIDTH){
                        count++;
                        x++;
                    }

                    y++;
                }

                width ++;
            }

            height += 3;
        }

        return count;
    }

    count = 0;

    while(height <= HEIGHT){
        width = minWidth;
        while(width <= WIDTH){
            int y = 0;

            while(y + height <= HEIGHT){
                int x = 0;

                while(x + width <= WIDTH){
                    init_feature_template(temps + count, x, y, width, height, HORIZONTAL_3);
                    count++;
                    x++;
                }

                y++;
            }

            width ++;
        }

        height += 3;
    }

    return count;
}


//CROSS
static int generate_feature_templates_type_cross(FeatTemp *temps, int WIDTH, int HEIGHT){
    int minWidth, height, width;
    int count;

    minWidth = 4;
    height = 4;
    width = 4;

    if(temps == NULL){
        count = 0;
        while(height <= HEIGHT){
            width = minWidth;
            while(width <= WIDTH){
                int y = 0;

                while(y + height <= HEIGHT){
                    int x = 0;

                    while(x + width <= WIDTH){
                        count++;
                        x++;
                    }

                    y++;
                }

                width += 4;
            }

            height += 4;
        }
        return count;
    }

    count = 0;

    while(height <= HEIGHT){
        width = minWidth;
        while(width <= WIDTH){
            int y = 0;

            while(y + height <= HEIGHT){
                int x = 0;

                while(x + width <= WIDTH){
                    init_feature_template(temps + count, x, y, width, height, CROSS);
                    count++;
                    x++;
                }

                y++;
            }

            width += 4;
        }

        height += 4;
    }

    return count;
}


int generate_feature_templates(int WINW, int WINH, FeatTemp **temps){
    int count[5] = {0};
    int num = 0;

    count[0] = generate_feature_templates_type_vertical_2(NULL, WINW, WINH);
    count[1] = generate_feature_templates_type_vertical_3(NULL, WINW, WINH);
    count[2] = generate_feature_templates_type_horizontal_2(NULL, WINW, WINH);
    count[3] = generate_feature_templates_type_horizontal_3(NULL, WINW, WINH);
    count[4] = generate_feature_templates_type_cross(NULL, WINW, WINH);

    num += count[0];
    num += count[1];
    num += count[2];
    num += count[3];
    num += count[4];

    printf("%d %d %d %d %d %d\n", count[0], count[1], count[2], count[3], count[4], num);

    *temps = new FeatTemp[num];

    generate_feature_templates_type_vertical_2(*temps, WINW, WINH);
    *temps += count[0];

    generate_feature_templates_type_vertical_3(*temps, WINW, WINH);
    *temps += count[1];

    generate_feature_templates_type_horizontal_2(*temps, WINW, WINH);
    *temps += count[2];

    generate_feature_templates_type_horizontal_3(*temps, WINW, WINH);
    *temps += count[3];

    generate_feature_templates_type_cross(*temps, WINW, WINH);
    *temps += count[4];

    return num;
}


