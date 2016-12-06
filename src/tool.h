#ifndef _TOOL_H_
#define _TOOL_H_

#include "typedef.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <assert.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int read_file_list(const char *filePath, std::vector<std::string> &fileList);
void analysis_file_path(const char* filePath, char *rootDir, char *fileName, char *ext);

void integral_image(uint8_t *img, int width, int height, int stride, uint32_t *intImg, int istride);
void resizer_bilinear_gray(uint8_t *src, int srcw, int srch, int srcs, uint8_t *dst, int dstw, int dsth, int dsts);
void update_weights(double *weights, int size);
#endif
