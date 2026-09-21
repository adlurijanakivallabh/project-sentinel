#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<16,6>, 1*1> input_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer18_t;
typedef ap_fixed<16,6> model_default_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer2_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer3_t;
typedef ap_fixed<18,8> conv2d_1_relu_table_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer4_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer5_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer19_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer6_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer7_t;
typedef ap_fixed<18,8> conv2d_2_relu_table_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer8_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer9_t;
typedef nnet::array<ap_fixed<16,6>, 448*1> layer10_t;
typedef nnet::array<ap_fixed<16,6>, 448*1> layer17_t;
typedef nnet::array<ap_fixed<16,6>, 128*1> layer11_t;
typedef nnet::array<ap_fixed<16,6>, 128*1> layer12_t;
typedef ap_fixed<18,8> conv1d_relu_table_t;
typedef nnet::array<ap_fixed<16,6>, 128*1> layer13_t;
typedef nnet::array<ap_fixed<16,6>, 128*1> layer14_t;
typedef nnet::array<ap_fixed<16,6>, 5*1> result_t;
typedef ap_uint<1> layer15_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
