#include "adaboost.h"


int sample_images(NegSetGenerator *generator){
    if(generator->imgList.empty() || generator->bufSize <= 0)
        return 1;

    cv::RNG rng(cv::getTickCount());
    int size = generator->imgList.size();

    generator->bufList.clear();

    for(int i = 0; i < generator->bufSize; i++){
        int id = rng.uniform(0, size);

        generator->bufList.push_back(generator->imgList[i]);

        HU_SWAP(generator->imgList[id], generator->imgList[size - 1], std::string);

        size --;
    }

    if(generator->imgs != NULL)
        delete [] generator->imgs;

    generator->imgs = new cv::Mat[generator->bufSize]; assert(generator->imgs != NULL);

    for(int i = 0; i < generator->bufSize; i++){
        cv::Mat img = cv::imread(generator->bufList[i], 0);
        if(img.cols > 720)
            cv::resize(img, img, cv::Size(720, img.rows * 720 / img.cols));

        img.copyTo(generator->imgs[i]);
        if(i % 1000 == 999)
            printf("%d ", i), fflush(stdout);
    }

    printf("\n");

    generator->id = 0;
    generator->scale = 3.0f;

    return 0;
}


void generate_negative_samples(Cascade *cc, NegSetGenerator *generator, SampleSet *negSet, int ssize){
    if(negSet->ssize > ssize)
        return ;

    int WINW = negSet->winw;
    int WINH = negSet->winh;
    int winSize = WINW * WINH;

    int num = ssize - negSet->ssize;
    int count = 0;

    cv::Mat img;

    int W = generator->scale * WINW;

    int dx = 0.3f * WINW;
    int dy = 0.3f * WINH;

    char rootDir[256], fileName[256], ext[30], filePath[256];

    reserve(negSet, ssize + 100);

    sample_images(generator);

    uint32_t *buffer = new uint32_t[2048 * 2048]; assert(buffer != NULL);
    uint32_t *intImg;
    float score;

    while(count < num){
        generator->imgs[generator->id].copyTo(img);

        if(generator->tflag){
            transform_image(img, WINW);
        }
        else{
            int H = W * img.rows / img.cols;
            H = HU_MAX(H, WINH);
            cv::resize(img, img, cv::Size(W, H));
        }

        int istride = img.cols + 1;

        analysis_file_path(generator->bufList[generator->id].c_str(), rootDir, fileName, ext);

        sprintf(filePath, "%s.jpg", fileName);

        intImg = buffer + img.cols + 1 + 1;

        memset(buffer, 0, sizeof(uint32_t) * (img.cols + 1) * (img.rows + 1));

        integral_image(img.data, img.cols, img.rows, img.step, intImg, istride);

        for(int y = 0; y <= img.rows - WINH; y += dy){
            for(int x = 0; x <= img.cols - WINW; x += dx){
                if( predict(cc, intImg + y * istride + x, istride, winSize, score) ){
                    create_sample(negSet->samples[negSet->ssize++], img.data + y * img.step + x, WINW, WINH, img.step, filePath);
                    count++;
                }
            }
        }
        printf("%d\r", count); fflush(stdout);

        generator->id ++;

        if((!generator->tflag) && generator->id >= generator->bufSize){
            generator->scale ++;

            if(generator->scale < 8){
                W = WINW * generator->scale;
            }
            else {
                generator->scale = 0;
                generator->tflag = 1;
            }

            generator->level ++;

            if(generator->level > 10)
                generator->tflag = 1;

            if(generator->level % 5 == 4)
                sample_images(generator);

            generator->id %= generator->bufSize;
        }
    }

    delete [] buffer;
}


int train(Cascade *cc, const char *posFilePath, const char *negFilePath){

    const int STAGE = 5;

    const int WINW = 96;
    const int WINH = 96;

    float recall = 0.99f;
    float precision[] = {0.5f, 0.75f, 0.9f, 0.95f, 0.99f};
    int npRatio[] = {1, 1, 1, 1, 1};

    SampleSet *posSet, *negSet, *valSet;
    NegSetGenerator *generator;

    posSet = NULL;
    negSet = new SampleSet; assert(negSet != NULL);
    valSet = new SampleSet; assert(valSet != NULL);

    generator = new NegSetGenerator; assert(generator != NULL);

    memset(negSet, 0, sizeof(SampleSet));
    memset(valSet, 0, sizeof(SampleSet));
    memset(generator, 0, sizeof(NegSetGenerator));

    read_samples(posFilePath, 0, WINW, WINH, &posSet); // read positive samples
    print_info(posSet, "pos ori");
    split(posSet, 0.3, valSet);                        // select validate samples for positive samples

    negSet->winw = WINW;
    negSet->winh = WINH;

    read_file_list(negFilePath, generator->imgList);

    generator->bufSize = 5000;
    generator->id = 0;
    generator->scale = 1;
    generator->level = 0;
    generator->tflag = 0;

    cc->WINW = WINW;
    cc->WINH = WINH;

    cc->sc = NULL;
    cc->ssize = 0;


    write_images(posSet, "log/pos");
    write_images(valSet, "log/val");

    print_info(posSet, "pos");
    print_info(valSet, "neg");

    int ret = system("mkdir -p model log");

    for(int i = STAGE - 1; i >= 0; i--){
        char filePath[256];
        sprintf(filePath, "cascade_%d.dat", i);
        if(load(cc, filePath) == 0){
            printf("LOAD MODEL %s SUCCESS\n", filePath);
            break;
        }
    }

    if(cc->ssize == 0){
        cc->sc = new StrongClassifier[STAGE]; assert(cc->sc != NULL);
        memset(cc->sc, 0, sizeof(StrongClassifier) * STAGE);
    }
    else {
        StrongClassifier *sc = new StrongClassifier[STAGE]; assert(cc->sc != NULL);
        memset(sc, 0, sizeof(StrongClassifier) * STAGE);

        memcpy(sc, cc->sc, sizeof(StrongClassifier) * cc->ssize);

        delete [] cc->sc;
        cc->sc = sc;
    }

    for(int i = cc->ssize; i < STAGE; i++){
        printf("---------------- CASCADE %d ----------------\n", i);
        printf("RECALL = %f, PRECISION = %f\n", recall, precision[i]);

        printf("GENERATE NAGATIVE SAMPLES\n");
        generate_negative_samples(cc, generator, negSet, posSet->ssize * npRatio[i]);

        //print_info(negSet, "neg");

        write_images(negSet, "log/neg");

        ret = train(cc->sc + i, posSet, negSet, valSet, recall, precision[i]);
        if(ret != 0) break;

        cc->ssize++;

        {
            char filePath[256];
            sprintf(filePath, "model/cascade_%d.dat", i);
            save(cc, filePath);
        }

        {
            char command[256];
            sprintf(command, "mv classifier.txt log/classifier_%d.txt", i);
            ret = system(command);
        }
        printf("---------------------------------------------\n");
    }

    release(&posSet);
    release(&negSet);
    release(&valSet);
    release(&generator);

    return ret;
}


int predict(Cascade *cc, uint32_t *iImg, int iStride, int winSize, float &score){
    float scores[10];

    score = 0;

    for(int i = 0; i < cc->ssize; i++){
        if(predict(cc->sc + i, iImg, iStride, winSize, scores[i]) == 0)
            return 0;
        score += scores[i];
    }

    return 1;
}


void init_detect_factor(Cascade *cc, float startScale, float endScale, float offset, int layer){
    cc->startScale = startScale;
    cc->endScale = endScale;
    cc->layer = layer;
    cc->offsetFactor = offset;
}


int calculate_max_size(int width, int height, float startScale, int winSize){
    int minwh = HU_MIN(width, height);

    assert(startScale < 1.0f);

    int size = minwh * startScale;
    float scale = (float)winSize / size;

    if(scale < 1)
        return width * height;

    return (width * scale + 0.5f) * (height * scale + 0.5f);
}


int merge_rect(HRect *rects, float *scores, int size){
    int8_t *flags = new int8_t[size]; assert(flags != NULL);

    memset(flags, 0, sizeof(int8_t) * size);

    for(int i = 0; i < size; i++){
        int xi0 = rects[i].x;
        int yi0 = rects[i].y;
        int xi1 = rects[i].x + rects[i].width - 1;
        int yi1 = rects[i].y + rects[i].height - 1;

        int cix = (xi0 + xi1) >> 1;
        int ciy = (yi0 + yi1) >> 1;
        int sqi = rects[i].width * rects[i].height;

        for(int j = i + 1; j < size; j++)
        {
            int xj0 = rects[j].x;
            int yj0 = rects[j].y;
            int xj1 = rects[j].x + rects[j].width - 1;
            int yj1 = rects[j].y + rects[j].height - 1;

            int cjx = (xj0 + xj1) >> 1;
            int cjy = (yj0 + yj1) >> 1;

            int sqj = rects[j].width * rects[j].height;

            bool acInB = (xi0 <= cjx && cjx <= xi1) && (yi0 <= cjy && cjy <= yi1);
            bool bcInA = (xj0 <= cix && cix <= xj1) && (yj0 <= ciy && ciy <= yj1);
            bool acNInB = (cjx < xi0 || cjx > xi1) || (cjy < yi0 || cjy > yi1);
            bool bcNInA = (cix < xj0 || cix > xj1) || (ciy < yj0 || ciy > yj1);

            if(acInB && bcInA){
                if(scores[j] > scores[i])
                    flags[i] = 1;
                else
                    flags[j] = 1;
            }
            else if(acInB && bcNInA){
                 flags[j] = 1;
            }
            else if(acNInB && bcInA){
                flags[i] = 1;
            }
        }
    }

    for(int i = 0; i < size; i++){
        if(flags[i] == 0) continue;

        HU_SWAP(rects[i], rects[size - 1], HRect);
        HU_SWAP(scores[i], scores[size - 1], float);

        size --;
        i --;
    }



    delete []flags;
    flags = NULL;

    return size;
}


int detect_one_scale(Cascade *cc, float scale, uint32_t *iImg, int width, int height, int stride, HRect *resRect, float *resScores){

    int WINW = cc->WINW;
    int WINH = cc->WINH;
    int winSize = WINW * WINH;

    int dx = WINW * cc->offsetFactor;
    int dy = WINH * cc->offsetFactor;

    int count = 0;
    float score;

    for(int y = 0; y <= height - WINH; y += dy){
        for(int x = 0; x <= width - WINW; x += dx){
            if(predict(cc, iImg + y * stride + x, stride, winSize, score) == 1){
                resRect[count].x = x * scale;
                resRect[count].y = y * scale;
                resRect[count].width = WINW * scale;
                resRect[count].height = WINH * scale;

                resScores[count] = score;
                count++;
            }
        }
    }

    if(count < 2)
        return count;

    count = merge_rect(resRect, resScores, count);
    return count;
}


int detect(Cascade *cc, uint8_t *img, int width, int height, int stride, HRect **resRect){
    int WINW, WINH, capacity;
    float scale, stepFactor;

    uint8_t *dImg, *ptrSrc, *ptrDst;
    uint32_t *iImgBuf, *iImg;

    HRect *rects;
    float *scores;
    int top = 0;

    int srcw, srch, srcs, dstw, dsth, dsts;
    int count;

    WINW = cc->WINW;
    WINH = cc->WINH;

    scale = cc->startScale;
    stepFactor = (cc->endScale - cc->startScale) / (cc->layer - 1);

    capacity = calculate_max_size(width, height, scale, HU_MAX(WINW, WINH));

    dImg = new uint8_t[capacity * 2]; assert(dImg != NULL);
    iImgBuf = new uint32_t[capacity * 2]; assert(iImgBuf != NULL);

    rects = new HRect[5000]; assert(rects != NULL);
    scores = new float[5000]; assert(scores != NULL);
    memset(rects, 0, sizeof(HRect) * 5000);
    memset(scores, 0, sizeof(float) * 5000);

    ptrSrc = img;
    ptrDst = dImg;

    srcw = width;
    srch = height;
    srcs = stride;

    count = 0;

    for(int i = 0; i < cc->layer; i++){
        dstw = scale * width;
        dsth = scale * height;
        dsts = dstw;

        resizer_bilinear_gray(ptrSrc, srcw, srch, srcs, ptrDst, dstw, dsth, dsts);

        memset(iImgBuf, 0, sizeof(uint32_t) * (dstw + 1) * (dsth + 1));
        iImg = iImgBuf + dstw + 1 + 1;

        integral_image(ptrDst, dstw, dsth, dsts, iImg, dstw + 1);

        count += detect_one_scale(cc, 1.0f / scale, iImg, dstw, dsth, dsts, rects + count, scores + count);

        ptrSrc = ptrDst;

        srcw = dstw;
        srch = dsth;
        srcs = dsts;

        if(ptrDst == dImg)
            ptrDst = dImg + dstw * dsth;
        else
            ptrDst = dImg;

        scale += stepFactor;
    }

    if(count > 0){
        if(count > 1)
            count = merge_rect(rects, scores, count);

        *resRect = new HRect[count]; assert(resRect != NULL);

        memcpy(*resRect, rects, sizeof(HRect) * count);
    }

    delete [] dImg;
    delete [] iImgBuf;
    delete [] rects;
    delete [] scores;

    return count;
}


int load(Cascade *cc, const char *filePath){
    if(cc == NULL)
        return 1;

    FILE *fin = fopen(filePath, "rb");
    if(fin == NULL)
        return 2;

    int ret;

    ret = fread(&cc->ssize, sizeof(int), 1, fin);

    cc->sc = new StrongClassifier[cc->ssize]; assert(cc->sc != NULL);

    for(int i = 0; i < cc->ssize; i++){
        ret = load(cc->sc + i, fin);
        if(ret != 0){
            fclose(fin);
            delete cc->sc;
            delete cc;
            return 2;
        }
    }

    fclose(fin);

    return 0;
}


int save(Cascade *cc, const char *filePath){
    FILE *fout = fopen(filePath, "wb");
    if(fout == NULL)
        return 1;

    int ret;

    ret = fwrite(&cc->ssize, sizeof(int), 1, fout);

    for(int i = 0; i < cc->ssize; i++){
        ret = save(cc->sc + i, fout);
        if(ret != 0){
            fclose(fout);
            return 2;
        }
    }

    fclose(fout);

    return 0;
}


void release_data(Cascade *cc){
    if(cc->sc != NULL)
        return;

    for(int i = 0; i < cc->ssize; i++)
        release_data(cc->sc);

    delete [] cc->sc;
    cc->sc = NULL;
}


void release(Cascade **cc){
    if(*cc == NULL)
        return;

    release_data(*cc);
    delete cc;
    cc = NULL;
}


void release_data(NegSetGenerator *ng){
    if(ng->imgs != NULL)
        delete [] ng->imgs;

    ng->imgs = NULL;
    ng->bufSize = 0;
}


void release(NegSetGenerator **ng){
    if(*ng != NULL){
        release_data(*ng);
        delete *ng;
    }

    *ng = NULL;
}
