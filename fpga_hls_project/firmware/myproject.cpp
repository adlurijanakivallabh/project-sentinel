#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    hls::stream<input_t> &input_layer,
    hls::stream<result_t> &layer15_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_layer,layer15_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<model_default_t, 288>(w2, "w2.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(s4, "s4.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b4, "b4.txt");
        nnet::load_weights_from_txt<model_default_t, 18432>(w6, "w6.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b6, "b6.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(s8, "s8.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b8, "b8.txt");
        nnet::load_weights_from_txt<model_default_t, 172032>(w11, "w11.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b11, "b11.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(s13, "s13.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b13, "b13.txt");
        nnet::load_weights_from_txt<model_default_t, 640>(w15, "w15.txt");
        nnet::load_weights_from_txt<model_default_t, 5>(b15, "b15.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=704

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=600

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=600

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=600

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=150

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS STREAM variable=layer19_out depth=204

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=150

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=150

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=150

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=35

    hls::stream<layer10_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=5

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=7

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=5

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=5

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=5

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=1

    nnet::zeropad2d_cl<input_t, layer18_t, config18>(input_layer, layer18_out); // zp2d_conv2d_1

    nnet::conv_2d_cl<layer18_t, layer2_t, config2>(layer18_out, layer2_out, w2, b2); // conv2d_1

    nnet::relu<layer2_t, layer3_t, relu_config3>(layer2_out, layer3_out); // conv2d_1_relu

    nnet::normalize<layer3_t, layer4_t, config4>(layer3_out, layer4_out, s4, b4); // bn2d_1

    nnet::pooling2d_cl<layer4_t, layer5_t, config5>(layer4_out, layer5_out); // pool2d_1

    nnet::zeropad2d_cl<layer5_t, layer19_t, config19>(layer5_out, layer19_out); // zp2d_conv2d_2

    nnet::conv_2d_cl<layer19_t, layer6_t, config6>(layer19_out, layer6_out, w6, b6); // conv2d_2

    nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out); // conv2d_2_relu

    nnet::normalize<layer7_t, layer8_t, config8>(layer7_out, layer8_out, s8, b8); // bn2d_2

    nnet::pooling2d_cl<layer8_t, layer9_t, config9>(layer8_out, layer9_out); // pool2d_2

    nnet::repack_stream<layer9_t, layer10_t, 2240>(layer9_out, layer16_out); // repack_reshape

    nnet::zeropad1d_cl<layer10_t, layer17_t, config17>(layer16_out, layer17_out); // zp1d_conv1d

    nnet::conv_1d_cl<layer17_t, layer11_t, config11>(layer17_out, layer11_out, w11, b11); // conv1d

    nnet::relu<layer11_t, layer12_t, relu_config12>(layer11_out, layer12_out); // conv1d_relu

    nnet::normalize<layer12_t, layer13_t, config13>(layer12_out, layer13_out, s13, b13); // bn1d

    nnet::global_pooling1d_cl<layer13_t, layer14_t, config14>(layer13_out, layer14_out); // global_pool

    nnet::dense<layer14_t, result_t, config15>(layer14_out, layer15_out, w15, b15); // fc

}

