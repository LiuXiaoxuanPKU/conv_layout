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

const int N = 1, H = 14, W = 14, IC = 128, OC = 256, KH = 3, KW = 3;
std::vector<mf> tags = {mf::nchw, mf::chwn, mf::nhwc, mf::nChw16c,
                        mf::nChw4c, mf::nChw8c, mf::NChw16n16c, mf::NChw32n32c};
std::vector<mf> wei_tags = {mf::oihw, mf::hwio, mf::ihwo};

double conv_format(engine::kind engine_kind,
                   mf input_f, // input memory format given by user
                   mf wei_f,
                   mf dst_f,
                   mf conv_input_f, // input memory format for conv
                   mf conv_wei_f,
                   mf conv_dst_f) {
  engine eng(engine_kind, 0);
  stream s(eng);

  std::clock_t start = std::clock();

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
  conv.execute(s, {{DNNL_ARG_SRC, conv_src_mem}, {DNNL_ARG_WEIGHTS, conv_weights_mem},
                   {DNNL_ARG_DST, conv_dst_mem}});
  s.wait();

  return std::clock() - start;
}

int main(int argc, char **argv) {
  double repeat_times = 10.0;
  int success_cnt = 0;
  for (int m = 0; m < (int) tags.size(); m++) {
    mf input_tag = tags[m];
    for (int n = 0; n < (int) wei_tags.size(); n++) {
      mf wei_tag = tags[n];
      for (int k = 0; k < (int) tags.size(); k++) {
        mf dst_tag = tags[k];
        double suboptimal_duration = 0; // conv on suboptimal layout specified by user
        double optimal_duration = 0; // reshape input + conv on optimal layout + reshape output
        for (int i = 0; i < repeat_times; i++) {
          suboptimal_duration += conv_format(parse_engine_kind(argc, argv),
                                             input_tag, wei_tag, dst_tag, input_tag, wei_tag, dst_tag);
        }

        for (int i = 0; i < repeat_times; i++) {
          optimal_duration += conv_format(parse_engine_kind(argc, argv),
                                          input_tag, wei_tag, dst_tag,
                                          mf::any,
                                          mf::any, mf::any);
        }
        std::cout << success_cnt << " " << m << " " << n << " " << k << " " << suboptimal_duration << " "
                  << optimal_duration << std::endl;
        if (suboptimal_duration < optimal_duration) {
          success_cnt += 1;
          std::cout << "[Success]" << m << " " << n << " " << k << std::endl;
        }
      }
    }
  }
}