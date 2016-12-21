#include "classifier.h"


static void init_feature_template(FeatTemp *ft, int x, int y, int w, int h, HaarType type){
    ft->x = x;
    ft->y = y;
    ft->w = w;
    ft->h = h;
    ft->type = type;
}


static int get_value(FeatTemp *temp, uint32_t *img, int stride, int winSize){
    uint16_t x = temp->x;
    uint16_t y = temp->y;
    uint16_t w = temp->w;
    uint16_t h = temp->h;

    winSize = 1;

    int x0, y0, x1, y1;

    int lef, rig, cen, top, bot, lt, rt, lb, rb;
    int len;
    int value;

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


            value = lef - rig;
            break;

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

            value = cen - lef - rig;
            break;

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

            value = top - bot;
            break;

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

            value = cen - top - bot;
            break;

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

            value = lt + rb - rt - lb;
            break;
    }

    return value / winSize;
}


static int generate_feature_templates(int WINW, int WINH, FeatTemp **temps);


#define FEATURE_NUM 2000

static void extract_features(SampleSet *set, FeatTemp *featTemps, int fsize, int *feats){
    int ssize = set->ssize;

    memset(feats, 0, sizeof(int) * fsize * ssize);

    for(int i = 0; i < ssize; i++){
        uint32_t *iImg = set->samples[i]->iImg;
        int istride =  set->samples[i]->istride;

        int *ptrFeats = feats + i;

        //printf("%5d: ", i);

        for(int f = 0; f < fsize; f++){
            //printf("%d ", f);
            ptrFeats[0] = get_value(featTemps + f, iImg, istride, set->winw * set->winh);
            ptrFeats += ssize;
        }

        //printf("\n");
    }
/*

    for(int j = 0; j < fsize; j++){
        int count = 0;
        for(int i = 0; i < ssize; i++){
            //printf("%d ", feats[j * ssize + i]);
            count += (feats[j * ssize + i] == 0) ;
        }

        if(count == ssize){
            print_feature_template(featTemps + j, stdout);
            printf("\n");
        }
    }
//*/
}

#define LENGTH 512

static void binary_classify_error(int *posFeats, double *pws, int posSize,
        int *negFeats, double *nws, int negSize,
        int &thresh, int8_t &sign, double &minError)
{
    double ptable[LENGTH], ntable[LENGTH];
    float minv = FLT_MAX, maxv = -FLT_MAX, step, istep;

    memset(ptable, 0, sizeof(double) * LENGTH);
    memset(ntable, 0, sizeof(double) * LENGTH);

    for(int i = 0; i < posSize; i++){
        minv = HU_MIN(minv, posFeats[i]);
        maxv = HU_MAX(maxv, posFeats[i]);
    }

    for(int i = 0; i < negSize; i++){
        minv = HU_MIN(minv, negFeats[i]);
        maxv = HU_MAX(maxv, negFeats[i]);
    }

    step = (maxv - minv) / (LENGTH - 1);
    istep = 1.0f / step;


    for(int i = 0; i < posSize; i++){
        int id = (posFeats[i] - minv) * istep;
        ptable[id] += pws[i];
    }

    for(int i = 0; i < negSize; i++){
        int id = (negFeats[i] - minv) * istep;
        ntable[id] += nws[i];
    }

    double nw = 0, pw = 0;
    double lw = 0, rw = 0;
    double error;

    minError = FLT_MAX;

    for(int i = 0; i < LENGTH; i++){
        pw += ptable[i];
        nw += ntable[i];

        lw = (pw + 1 - nw) / 2.0;
        rw = 1.0 - lw;

        error = HU_MIN(lw, rw);

        if(error < minError){
            minError = error;
            thresh = minv + i * step;
            sign = (pw < 0.5);
            //sign = 0, pos: > thresh
            //sign = 1, pos: <= thresh
        }
    }
}



static void split(int *posFeats, double *pws, float *pss, int posSize,
        int *negFeats, double *nws, float *nss, int negSize,
        int bestThresh, int bestSign, float score, float beta){

    double tp = 0.0f, fp = 0.0f;
    double fn = 0.0f, tn = 0.0f;

    if(bestSign == 0){
        for(int i = 0; i < posSize; i++){
            if(posFeats[i] > bestThresh){
                tp += pws[i];

                pws[i] *= beta;
                pss[i] += score;
            }
            else {
                fn += pws[i];
                pss[i] -= score;
            }
        }

        for(int i = 0; i < negSize; i++){
            if(negFeats[i] <= bestThresh){
                tn += nws[i];

                nws[i] *= beta;
                nss[i] -= score;
            }
            else {
                fp += nws[i];
                nss[i] += score;
            }
        }
    }
    else {
        for(int i = 0; i < posSize; i++){
            if(posFeats[i] <= bestThresh){
                tp += pws[i];
                pws[i] *= beta;
                pss[i] += score;
            }
            else {
                fn += pws[i];

                pss[i] -= score;
            }
        }

        for(int i = 0; i < negSize; i++){
            if(negFeats[i] > bestThresh){
                tn += nws[i];

                nws[i] *= beta;
                nss[i] -= score;
            }
            else {
                fp += nws[i];
                nss[i] += score;
            }
        }
    }

    printf("TP %f, FP %f, FN %f, TN %f, recall %f, precision %f\n", tp, fp, fn, tn, tp / (tp + fn), tp / (tp + fp));
}


static void expand_space(StrongClassifier *sc){
    int len = 20;

    if(sc->capacity == 0){
        sc->capacity = len;
        sc->ssize = 0;

        sc->featTemps = new FeatTemp[sc->capacity];
        sc->wcss = new float[sc->capacity];
        sc->wcts = new int[sc->capacity];
        sc->signs = new int8_t[sc->capacity];

        sc->thresh = 0;

        return ;
    }

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
        delete [] sc->signs;
        sc->signs = signs;
    }
}


static float get_precision(StrongClassifier *sc, SampleSet *posSet, SampleSet *negSet, float recall, float &thresh){
    int posSize = posSet->ssize;
    int negSize = negSet->ssize;

    int winSize = posSet->winw * posSet->winh;

    int fsize = sc->ssize;


    float *posScores = new float[posSize];
    float *negScores = new float[negSize];

    memset(posScores, 0, sizeof(float) * posSize);
    memset(negScores, 0, sizeof(float) * negSize);

    for(int i = 0; i < posSize; i++){
        Sample *sample = posSet->samples[i];
        predict(sc, sample->iImg, sample->istride, winSize, posScores[i]);
    }

    for(int i = 0; i < negSize; i++){
        Sample *sample = negSet->samples[i];
        predict(sc, sample->iImg, sample->istride, winSize, negScores[i]);
    }

    sort_arr_float(posScores, posSize);
    sort_arr_float(negScores, negSize);

    recall = 1.0f - recall;
    printf("%f %d\n", recall, int(recall * posSize));

    thresh = posScores[int(recall * posSize)] - FLT_EPSILON;

    int pID = 0;
    for( ; pID < posSize; pID++){
        if(posScores[pID] > thresh)
            break;
    }

    int nID = 0;
    for(; nID < negSize; nID++){
        if(negScores[nID] > thresh)
            break;
    }

    delete [] posScores;
    delete [] negScores;

    float TP = 1.0f - float(pID) / posSize;
    float FP = 1.0f - float(nID) / negSize;

    printf("STRONG CLASSIFIER: %f %f\n", TP, FP);

    return (TP) / (TP + FP);
}


void print_feature_template(FeatTemp *ft, FILE *fout){
    fprintf(fout, "Feature Template: %d %d %d %d ", ft->x, ft->y, ft->w, ft->h);

    switch(ft->type){
        case VERTICAL_2:
            fprintf(fout, "VERTICAL 2");
            break;

        case VERTICAL_3:
            fprintf(fout, "VERTICAL 3");
            break;

        case HORIZONTAL_2:
            fprintf(fout, "HORIZONTAL 2");
            break;

        case HORIZONTAL_3:
            fprintf(fout, "HORIZONTAL 3");
            break;

        case CROSS:
            fprintf(fout, "CROSS");
            break;
    }

    fprintf(fout, "\n");
}


int train(StrongClassifier *sc, SampleSet *posSet, SampleSet *negSet, SampleSet *valSet, float recall, float precision){
    float rate = 0.0f;

    int WINW = posSet->winw;
    int WINH = posSet->winh;

    FeatTemp *featTemps = NULL;
    FeatTemp *featTemps2 = NULL;
    int fsize;

    printf("GENERATE FEATURE TEMPLATE\n");

    fsize = generate_feature_templates(WINW, WINH, &featTemps); assert(featTemps != NULL);
    featTemps2 = new FeatTemp[FEATURE_NUM]; assert(featTemps2 != NULL);

    assert(fsize > 0);

    int posSize = posSet->ssize;
    int negSize = negSet->ssize;
    int valSize = valSet->ssize;

    int *posFeats = new int[posSize * FEATURE_NUM]; assert(posFeats != NULL);
    int *negFeats = new int[negSize * FEATURE_NUM]; assert(negFeats != NULL);

    double *pws = new double[posSize]; assert(pws != NULL);
    double *nws = new double[negSize]; assert(nws != NULL);

    float *pss = new float[posSize]; assert(pss != NULL);
    float *nss = new float[negSize]; assert(nss != NULL);

    cv::RNG rng(cv::getTickCount());

    for(int i = 0; i < posSize; i++){
        pws[i] = 1.0;
        pss[i] = 0.0f;
    }

    for(int i = 0; i < negSize; i++){
        nws[i] = 1.0;
        nss[i] = 0.0f;
    }

    printf("CREATE CLASSIFIER\n");
    FILE *flog = fopen("classifier.txt", "w");
    int cId = 0;

    printf("TRAIN WEAK CLASSIFIER\n");

    while(rate < precision){
        double minError;
        int8_t bestSign;

        int bestThresh;
        int bestID;

        printf("%d ", cId); fflush(stdout);
        fprintf(flog, "classifier: %d\n", cId ++);

        expand_space(sc);

        memset(featTemps2, 0, sizeof(FeatTemp) * FEATURE_NUM);

        for(int i = 0; i < FEATURE_NUM; i++){
            int id = rng.uniform(0, fsize - i);

            featTemps2[i] = featTemps[id];

            HU_SWAP(featTemps[fsize - i - 1], featTemps[id], FeatTemp);
        }

        extract_features(posSet, featTemps2, FEATURE_NUM, posFeats);
        extract_features(negSet, featTemps2, FEATURE_NUM, negFeats);

        update_weights(pws, posSize);
        update_weights(nws, negSize);

        bestID = -1;
        bestThresh = 0;
        minError = FLT_MAX;
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

        printf("minError: %f\n", minError);
        printf("id: %d, %d %d %d %d type: %d\n", bestID, featTemps2[bestID].x,featTemps2[bestID].y,
                featTemps2[bestID].w,featTemps2[bestID].h,featTemps2[bestID].type);


        fprintf(flog, "minError: %f\n", minError);
        fprintf(flog, "id: %d, %d %d %d %d type: %d\n", bestID, featTemps2[bestID].x,featTemps2[bestID].y,
                featTemps2[bestID].w,featTemps2[bestID].h,featTemps2[bestID].type);

        double beta = minError / (1 - minError);
        double score = log(1 / beta);

        int *bestPosFeats = posFeats + bestID * posSize;
        int *bestNegFeats = negFeats + bestID * negSize;

        FeatTemp bestTemp = featTemps2[bestID];

        split(posFeats, pws, pss, posSize, negFeats, nws, nss, negSize, bestThresh, bestSign, score, beta);

        sc->featTemps[sc->ssize] = featTemps2[bestID];
        sc->wcts[sc->ssize] = bestThresh;
        sc->signs[sc->ssize] = bestSign;

        if(bestSign == 0){
            int id = sc->ssize << 1;

            sc->wcss[id] = -score;
            sc->wcss[id + 1] = score;
        }
        else {
            int id = sc->ssize << 1;

            sc->wcss[id] = score;
            sc->wcss[id + 1] = -score;
        }

        sc->ssize ++;

        printf("Calculate precision\n");
        rate = get_precision(sc, valSet, negSet, recall, sc->thresh);

        fprintf(flog, "sc thresh: %f precision: %f\n\n", sc->thresh, rate); fflush(flog);
        printf("sc thresh: %f precision: %f\n\n", sc->thresh, rate); fflush(flog);
    }

    fclose(flog);

    delete [] featTemps, delete [] featTemps2;
    delete [] posFeats, delete [] negFeats;
    delete [] pws, delete [] nws;
    delete [] pss, delete [] nss;

    return 0;
}


int predict(StrongClassifier *sc, uint32_t *intImg, int istride, int winSize, float &score){
    score = 0;

    for(int i = 0; i < sc->ssize; i++){
        int value = get_value(sc->featTemps + i, intImg, istride, winSize);
        score += sc->wcss[(i << 1) + (value > sc->wcts[i])];
    }

    return (score > sc->thresh);
}


int save(StrongClassifier *sc, FILE *fout){
    if(fout == NULL || sc == NULL)
        return 1;
    int ret;

    ret = fwrite(&sc->ssize, sizeof(int), sc->ssize, fout);
    ret = fwrite(sc->featTemps, sizeof(FeatTemp), sc->ssize, fout);
    ret = fwrite(sc->signs, sizeof(int8_t), sc->ssize, fout);
    ret = fwrite(sc->wcts, sizeof(int), sc->ssize, fout);

    for(int i = 0; i < sc->ssize; i++){
        float score = fabs(sc->wcss[i << 1]);

        fwrite(&score, sizeof(float), 1, fout);
    }

    return 0;
}


int load(StrongClassifier *sc, FILE *fin){
    if(fin == NULL || sc == NULL){
        return 1;
    }

    int ret;

    ret = fread(&sc->ssize, sizeof(int), 1, fin);

    if(ret != 1) return 2;

    sc->featTemps = new FeatTemp[sc->ssize]; assert(sc->featTemps != NULL);
    sc->wcss = new float[sc->ssize << 1]; assert(sc->wcss != NULL);
    sc->wcts = new int[sc->ssize]; assert(sc->wcts != NULL);
    sc->signs = new int8_t[sc->ssize]; assert(sc->signs != NULL);
    sc->capacity = sc->ssize;


    ret = fread(sc->featTemps, sizeof(FeatTemp), sc->ssize, fin);
    if(ret != sc->ssize)
        return 3;

    ret = fread(sc->signs, sizeof(int8_t), sc->ssize, fin);
    if(ret != sc->ssize) return 4;

    ret = fread(sc->wcts, sizeof(int), sc->ssize, fin);
    if(ret != sc->ssize) return 5;

    for(int i = 0; i < sc->ssize; i++){
        float score;

        ret = fread(&score, sizeof(float), 1, fin);
        if(ret != 1) return 6;

        if(sc->signs[i] == 1){
            int id = i << 1;
            sc->wcss[id] = -score;
            sc->wcss[id + 1] = score;
        }
        else {
            int id = i << 1;
            sc->wcss[id] = score;
            sc->wcss[id + 1] = -score;
        }
    }

    return 0;
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


static int generate_feature_templates(int WINW, int WINH, FeatTemp **temps){
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

    *temps = new FeatTemp[num]; assert(*temps != NULL);

    if(generate_feature_templates_type_vertical_2(*temps, WINW, WINH) != count[0])
        return 0;

    *temps += count[0];

    if(generate_feature_templates_type_vertical_3(*temps, WINW, WINH) != count[1])
        return 0;

    *temps += count[1];

    if(generate_feature_templates_type_horizontal_2(*temps, WINW, WINH) != count[2])
        return 0;

    *temps += count[2];

    if(generate_feature_templates_type_horizontal_3(*temps, WINW, WINH) != count[3])
        return 0;

    *temps += count[3];

    if(generate_feature_templates_type_cross(*temps, WINW, WINH) != count[4])
        return 0;

    *temps += count[4];

    *temps -= num;
    return num;
}


void release(StrongClassifier **sc){
    if(sc == NULL)
        return;

    release_data(*sc);

    delete *sc;
    *sc = NULL;
}


void release_data(StrongClassifier *sc){
    if(sc != NULL){
        if(sc->featTemps != NULL)
            delete [] sc->featTemps;
        sc->featTemps = NULL;

        if(sc->wcss != NULL)
            delete [] sc->wcss;

        sc->wcss = NULL;

        if(sc->wcts != NULL)
            delete [] sc->wcts;

        sc->wcts = NULL;

        if(sc->signs != NULL)
            delete  [] sc->signs;

        sc->signs = NULL;
    }

    sc->ssize = 0;
    sc->capacity = 0;
}
