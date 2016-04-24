#include "adaboost.h"


FILE *flog = fopen("log/log.txt", "w");


void sampling(int size, int sampleSize, std::vector<int> &res){
    cv::RNG rng(cv::getTickCount());
    std::vector<int> idxs;

    for(int i = 0; i < size; i++){
        idxs.push_back(i);
    }

    for(int i = 0; i < sampleSize; i++){
        int id = rng.uniform(0, size);

        res.push_back(idxs[id]);

        HU_SWAP(idxs[id], idxs[size - 1], int);

        size --;
    }
}


int generate_negative_samples(std::vector<std::string> &imgList, int WIDTH, int HEIGHT, CascadeClassifier *cc, std::vector<Sample*> &negativeSet, int size)
{
    const float MAX_SCALE_TIMES = 10.0;
    static float sc = 1.0;

    int dx = 0.4 * WIDTH;
    int dy = 0.4 * HEIGHT;

    std::vector<int> idxs;
    int w, h;

    int id = 0;
    int startCount = 0, count = 0;
    int needNum = size - negativeSet.size();

    std::string imgPath;
    cv::Mat img;

    sampling(imgList.size(), 0.5 * imgList.size(), idxs);

    int NEG_IDX = 0;
    char outname[128];

    const int STRIDE = ((WIDTH + 3) >> 2) << 2;
    Sample *sample = create_sample(NULL, WIDTH, HEIGHT, STRIDE);

    while(count < needNum){
        imgPath = imgList[idxs[id]];
        img = cv::imread(imgPath, 0);

        w = sc * WIDTH;
        h = img.rows * w / img.cols;
        h = HU_MAX(HEIGHT, h);

        cv::resize(img, img, cv::Size(w, h));

        /*
        sprintf(outname, "log/neg/%d.jpg", NEG_IDX++);
        cv::imwrite(outname, img);
        */

        w -= WIDTH;
        h -= HEIGHT;

        startCount = count;

        for(int y = 0; y <= w; y += dy){
            for(int x = 0; x <= h; x += dx){
                set_image(sample, img.data + y * img.step + x, WIDTH, HEIGHT, img.step);

                if(classify(cc, sample) == 1){
                    negativeSet.push_back(sample);
                    sample = create_sample(NULL, WIDTH, HEIGHT, STRIDE);
                    count++;
                }
            }
        }

        id++;
        if(id >= idxs.size()){
            if(sc < MAX_SCALE_TIMES)
                sc += 0.5;
            else {
                printf("GET TO THE END\n");
                sampling(imgList.size(), 1.0 * imgList.size(), idxs);
                sc = 1.0;
                dx = 0.2 * WIDTH;
                dy = 0.2 * HEIGHT;
            }

            id = 0;
            printf("SCALE FACTOR: %f\n", sc);
        }

        if(startCount < count){
            printf("%.2f\r", count * 100.0 / needNum); fflush(stdout);
        }
    }

    release_sample(&sample);
    printf("Negset size: %ld\n", negativeSet.size());

    return 0;
}


void select_feature(std::vector<Feature*> &featureSet, int featDim, std::vector<Feature*> &fineFeats){
    //printf("SELECT FEATURE\n");
    cv::RNG rng(cv::getTickCount());

    int size = featureSet.size();
    std::vector<int> idxs(size);

    fineFeats.resize(featDim);

    for(int i = 0; i < size; i++)
        idxs[i] = i;

    assert(size > featDim);

    for(int i = 0; i < featDim; i++){
        int id = rng.uniform(0, size);

        fineFeats[i] = featureSet[idxs[id]];

        idxs[id] = idxs[size - 1];
        size --;
    }

    assert(size > 0);
}


void adaboost_learning(CascadeClassifier *cc, std::vector<Sample*> &posSet, std::vector<Sample*> &negSet, std::vector<Sample*> &valSet,
        std::vector<Feature*> featSet, float maxfpr, float maxfnr){
    StrongClassifier * sc = NULL;

    int numPos = posSet.size();
    int numNeg = negSet.size();

    int sampleSize = numPos + numNeg;
    float *weights = NULL, *values = NULL;

    const int FEATURE_SIZE = 4000;

    float cfpr = 1.0;

    sc = new StrongClassifier;
    memset(sc, 0, sizeof(StrongClassifier));

    init_weights(&weights, numPos, numNeg);

    assert(sampleSize > 2);

    values = new float[sampleSize];
    memset(values, 0, sizeof(float) * sampleSize);

    while(cfpr > maxfpr){
        std::vector<Feature*> fineFeatSet;
        float minError = 1.0, error, beta;

        int tp, tn;

        WeakClassifier *bestWC = new WeakClassifier;
        WeakClassifier *wc = new WeakClassifier;

        init_weak_classifier(bestWC, 0, 0, NULL);
        init_weak_classifier(wc, 0, 0, NULL);

        bestWC->feat = new Feature;
        wc->feat = new Feature;

        select_feature(featSet, FEATURE_SIZE, fineFeatSet);

        for(int i = 0; i < FEATURE_SIZE; i++){
            Feature *feat = fineFeatSet[i];

            init_weak_classifier(wc, 0, 0, wc->feat);
            init_feature(wc->feat, feat);

            extract_feature_values(feat, posSet, values);
            extract_feature_values(feat, negSet, values + numPos);

            error = train(wc, values, numPos, numNeg, weights);

            if(error < minError){
                minError = error;
                if(error < 0.01){
                    printf("feat:%d %d %d %d %d, minError = %f\n", feat->x0, feat->y0, feat->w, feat->h, feat->type, minError);

                    write_samples(posSet, "log/pos");
                    write_samples(negSet, "log/neg");
                    exit(0);
                }

                HU_SWAP(wc, bestWC, WeakClassifier*);

                printf("Select best weak classifier, min error: %f\r", minError); fflush(stdout);
            }
        }


        fineFeatSet.clear();
        clear(&wc);

        assert(minError > 0);

        printf("best weak classifier error = %f                      \n", minError);

        beta = minError / (1 - minError);

        tp = 0;

        for(int i = 0; i < numPos; i++ ){
            if(classify(bestWC, posSet[i]) == POS_SAMPLE_FLAG){
                weights[i] *= beta;
                tp ++;
            }
        }

        tn = 0;
        for(int i = 0, j = numPos; i < numNeg; i++, j++){
            if(classify(bestWC, negSet[i]) == NEG_SAMPLE_FLAG){
                weights[j] *= beta;
                tn++;
            }
        }

        printf("TP = %d, TN = %d, beta = %f, log(1/beta) = %f\n", tp, tn, beta, log(1/beta));
        update_weights(weights, numPos, numNeg);

        add(sc, bestWC, log(1/beta));

        train(sc, posSet, maxfnr);

        cfpr = fpr(sc, valSet);

        printf("fpr validate: %f, fpr negative: %f\n\n", cfpr, fpr(sc, negSet));
    }

    printf("weak classifier size: %ld\n", sc->wcs.size());

    delete [] weights;
    delete [] values;

    add(cc, sc);
}


void detect_object(CascadeClassifier *cc, cv::Mat &img, float startScale, float endScale, int layers, float offsetFactor, std::vector<cv::Rect> &rects)
{
    cv::Mat gray, sImg;

    const int WINW = cc->WIDTH;
    const int WINH = cc->HEIGHT;
    const int STRIDE = ((WINW + 3) >> 2) << 2;


    int dx = offsetFactor * WINW;
    int dy = offsetFactor * WINH;

    float scaleStep = (endScale - startScale) / layers;

    if(endScale > startScale) {
        startScale = endScale;
        scaleStep = -scaleStep;
    }

    if(img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img.clone();

    Sample *sample = create_sample(NULL, WINW, WINH, STRIDE);

    for(int i = 0; i < layers; i++){
        cv::resize(gray, sImg, cv::Size(startScale * gray.cols, startScale * gray.rows));

        int ws = sImg.cols - WINW;
        int hs = sImg.rows - WINH;

        for(int y = 0; y <= hs; y += dy)
        {
            for(int x = 0; x <= ws; x += dx)
            {
                cv::Rect rect = cv::Rect(x, y, WINW, WINH);
                cv::Mat patch(sImg, rect);

                set_image(sample, patch.data, WINW, WINH, patch.step);

                if(classify(cc, sample) == 1)
                {
                    rect.x /= startScale;
                    rect.y /= startScale;

                    rect.width /= startScale;
                    rect.height /= startScale;

                    rects.push_back(rect);
                }
            }

        }

        startScale += scaleStep;
    }

    release_sample(&sample);
}
