#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxWithRebalancedLossLayer<Dtype>::LayerSetUp(
   const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  labels_.Reshape(bottom[1]->num(), 1, 1, 1);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxWithRebalancedLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, &softmax_top_vec_);
  if (top->size() >= 2) {
    // softmax output
    (*top)[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithRebalancedLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  // what is _prob ? looks like instantiation of some class
  // softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  const int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  //prob_.count() := no entries in prob_data ?
  //prob_.num()   := batchsize ?
  //dim           := no classes ?
  //spatial_dim   := 1 ?
  assert (spatial_dim == 1);

  float prior[dim];
  std::fill_n(prior, dim, 0);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      prior[static_cast<int>(label[i*spatial_dim+j])] += 1.0 / num;
    }
  }
  
  Dtype loss = 0;
  
  // std::cout << "Yes hello this is softmax rebalanced" << std::endl;

  for (int i = 0; i < num; ++i) {
  // std::cout << "loss: ";
    //why the fuck 2 loops ?
    //what's this crazy indexing of prob_data ??
    //oh... unless spatial_dim == 1 ! then it's as before
    for (int j = 0; j < spatial_dim; j++) {
      loss -= log(max(prob_data[i * dim + static_cast<int>(
				label[i * spatial_dim + j])
				* spatial_dim + j]
		      / (dim *
			 prior[static_cast<int>(
			       label[i * spatial_dim + j])]),
                      Dtype(FLT_MIN)));
    }
    // std::cout << loss << ", ";
  }
  
  // std::cout << std::endl;
  (*top)[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
  if (top->size() == 2) {
    (*top)[1]->ShareData(prob_);
  }
}

template <typename Dtype>
// computes dE/dz for every neuron input vector z = <x,w>+b
// this does NOT update the weights, it merely calculates dy/dz
void SoftmaxWithRebalancedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // Compute the diff
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = prob_.num();         //batchSize, num imgs
    const int dim = prob_.count() / num; //num neurons
    int spatial_dim = prob_.height() * prob_.width();

        
    float prior[dim];
    std::fill_n(prior, dim, 0);
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; j++) {
	prior[static_cast<int>(label[i*spatial_dim+j])] += 1.0 / num;
      }
    }

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        bottom_diff[i * dim + static_cast<int>(
		    label[i*spatial_dim + j])
                    * spatial_dim + j] -= 1;
      }
    }
    // Scale down gradient
    caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
  
    for (int k = 0; k < dim; ++k) {
      for (int i = 0; i < num; ++i)
	for (int j = 0; j < spatial_dim; j++) {
	  if (prior[static_cast<int>(label[i*spatial_dim+j])] > 0) {
	    // std::cout << bottom_diff[i * dim + static_cast<int>(label[i*spatial_dim + j])] << " / (" << std::endl;
	    // std::cout << bottom_diff[i * dim + static_cast<int>(label[i*spatial_dim + j])] << " / (" << prior[static_cast<int>(label[i*spatial_dim+j]] << " * " << dim << ")" << std::endl;
				     
	    bottom_diff[i * dim + static_cast<int>(
		    label[i*spatial_dim + j])* spatial_dim + j] /=		 (prior[static_cast<int>(label[i*spatial_dim+j])] * dim);
	  }
	}
    }
  }
}

INSTANTIATE_CLASS(SoftmaxWithRebalancedLossLayer);


}  // namespace caffe
