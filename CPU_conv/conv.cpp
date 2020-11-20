//
// Created by Xiaoxuan Liu on 11/3/20.
//

#include <iostream>
#include <ctime>

#include "dnnl.hpp"
#include "example_utils.hpp"

using namespace dnnl;
typedef memory::format_tag mf;

// Try to find cases where
// conv on suboptimal layout < reshape + conv on optimal layout

const int N = 1, H = 200, W = 200, IC = 3, OC = 3, KH = 3, KW = 3;
std::vector<mf> tags = {mf::nchw};
std::vector<mf> wei_tags = {mf::nchw};

double conv_format(engine::kind engine_kind,
                   mf input_f, // input memory format given by user
                   mf wei_f,
                   mf dst_f,
                   mf conv_input_f, // input memory format for conv
                   mf conv_wei_f,
                   mf conv_dst_f,
                   int repeat_times) {
  engine eng(engine_kind, 0);
  stream s(eng);


  auto conv_src_md = memory::desc({N, IC, H, W}, memory::data_type::f32, conv_input_f);
  auto conv_weights_md = memory::desc({OC, IC, KH, KW}, memory::data_type::f32, conv_wei_f);
  auto conv_dst_md = memory::desc({N, OC, H, W}, memory::data_type::f32, conv_dst_f);

  auto conv_pd = convolution_forward::primitive_desc(
      {prop_kind::forward_inference, algorithm::convolution_auto,
       conv_src_md, conv_weights_md, conv_dst_md,
       {1, 1}, // strides
       {1, 1}, {1, 1}, // left and right padding
      }, eng
  );

  memory conv_src_mem;
  memory conv_weights_mem;
  memory conv_dst_mem;
  auto src_mem
      = memory({{N, IC, H, W}, memory::data_type::f32, input_f}, eng);
  auto weights_mem = memory(
      {{OC, IC, KH, KW}, memory::data_type::f32, wei_f}, eng);
  auto dst_mem
      = memory({{N, OC, H, W}, memory::data_type::f32, dst_f}, eng);

  bool need_reorder_src = conv_pd.src_desc() != src_mem.get_desc();
  //printf("Need reorder src %d\n", need_reorder_src);
  bool need_reorder_weights
      = conv_pd.weights_desc() != weights_mem.get_desc();
  //printf("Need reorder weights %d\n", need_reorder_weights);
  bool need_reorder_dsts = conv_pd.dst_desc() != dst_mem.get_desc();
  //printf("Need reorder destination %d\n", need_reorder_dsts);

  conv_src_mem
      = need_reorder_src ? memory(conv_pd.src_desc(), eng) : src_mem;
  conv_weights_mem = need_reorder_weights
                     ? memory(conv_pd.weights_desc(), eng)
                     : weights_mem;
  conv_dst_mem
      = need_reorder_dsts ? memory(conv_pd.dst_desc(), eng) : dst_mem;

  if (need_reorder_src) {
    auto reorder_src = reorder(src_mem, conv_src_mem);
    reorder_src.execute(
        s, {{DNNL_ARG_FROM, src_mem}, {DNNL_ARG_TO, conv_src_mem}});
    s.wait();
  }
  if (need_reorder_weights) {
    auto reorder_weights = reorder(weights_mem, conv_weights_mem);
    reorder_weights.execute(s,
                            {{DNNL_ARG_FROM, weights_mem},
                             {DNNL_ARG_TO, conv_weights_mem}});
    s.wait();
  }

  if (need_reorder_dsts) {
    auto reorder_dst = reorder(conv_dst_mem, dst_mem);
    reorder_dst.execute(
        s, {{DNNL_ARG_FROM, conv_dst_mem}, {DNNL_ARG_TO, dst_mem}});
    s.wait();
  }

  auto conv_scratchpad_mem = memory(conv_pd.scratchpad_desc(), eng);
  
  auto conv = convolution_forward(conv_pd);

  std::clock_t start = std::clock();
  for (int i = 0; i < repeat_times; i++) {
    conv.execute(s, {{DNNL_ARG_SRC, conv_src_mem}, {DNNL_ARG_WEIGHTS, conv_weights_mem},
                    {DNNL_ARG_DST, conv_dst_mem}});
    s.wait();
  }

  double throughput = N * IC * H * W * sizeof(float) * repeat_times * 1e-9 / ((std::clock() - start) * 1.0 / CLOCKS_PER_SEC); 
  return throughput;
}

int main(int argc, char **argv) {
  int repeat_times = 1000;
  mf input_tag = mf::nchw;
  mf dst_tag = mf::nchw;
  mf wei_tag = mf::nhwc;
  double throughput = conv_format(parse_engine_kind(argc, argv),
                                          input_tag, wei_tag, dst_tag,
                                          mf::any,
                                          mf::any, mf::any, repeat_times);
  std::cout << "[CPU]Run " << repeat_times 
            << " conv, throughput " << throughput << " GB/s\n";
}