#ifndef _ADA_BOOST_H_
#define _ADA_BOOST_H_

#include "tool.h"
#include "cascade_classifier.h"


void adaboost_learning(CascadeClassifier *cc, std::vector<Sample*> &posSet, std::vector<Sample*> &negSet, std::vector<Sample*> &valSet,
        std::vector<Feature*> featSet, float maxfpr, float maxfnr);

void detect_object(CascadeClassifier *cc, cv::Mat &img, float startScale, float endScale, int layers, float offsetFactor, std::vector<cv::Rect> &rects);

int generate_negative_samples(std::vector<std::string> &imgList, int WIDTH, int HEIGHT, CascadeClassifier *cc, std::vector<Sample*> &negativeSet, int size);

#endif
