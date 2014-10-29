// Copyright 2014 BVLC and contributors.
// #include <iostream>
#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <cfloat>
#include <iostream>
// #include <cmath>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {
  
template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
}

template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 1, 1, 1);
  int dim = bottom[0]->count() / bottom[0]->num();
  int accuracies_count = dim + 2;
  (*top)[0]->Reshape(1, 1, 1, accuracies_count);
  accuracies_.Reshape(1, 1, 1, accuracies_count);    
  labels_count_.Reshape(1, 1, 1, dim);  
}


template <typename Dtype>
void PerClassAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data(); //threshold_layer calls this bottom_data
  // Dtype accuracy = 0;
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  int accuracies_count = dim + 2;

  Dtype* labels_count = labels_count_.mutable_cpu_data();
  Dtype* accuracies = accuracies_.mutable_cpu_data();
  caffe_set(labels_count_.count(), Dtype(FLT_MIN), labels_count);
  caffe_set(accuracies_.count(), Dtype(FLT_MIN), accuracies);
  
  for (int i = 0; i < num; ++i) {
    //count freq of each class
    labels_count[static_cast<int>(bottom_label[i])] += 1.0;
    //determine whether correctly classified
    Dtype maxval = -FLT_MAX;
    int max_id = 0;
    for (int j = 0; j < dim; ++j) {
      //find which class gets highest prob
      if (bottom_data[i * dim + j] > maxval) {
        maxval = bottom_data[i * dim + j];
        max_id = j;
      }
    }
    if (max_id == static_cast<int>(bottom_label[i]))
      accuracies[static_cast<int>(bottom_label[i])] += 1.0;
  }

  for (int j = 0; j < dim; ++j) {
    //accuracy averaged across cases
    accuracies[accuracies_count-1] += accuracies[j];
    //accuracy for class j
    accuracies[j] /= static_cast<float>(labels_count[j]);
    //accuracy averaged across classes
    accuracies[accuracies_count-2] += accuracies[j];
  }
  accuracies[accuracies_count-1] /= static_cast<float>(num);
  accuracies[accuracies_count-2] /= static_cast<float>(dim);

  // std::cout << "Accuracies: ";
  // for (int j = 0; j < accuracies_count; ++j) {
  //   std::cout << accuracies[j] << ", ";  
  // }
  // std::cout << std::endl;
  
  // LOG(INFO) << "Accuracies, class by class: " << accuracy;
  //can I do this or does 
  caffe_copy(accuracies_count, accuracies, (*top)[0]->mutable_cpu_data());
  // (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  // (*top)[0]->mutable_cpu_data()[0] = 1;
  // (*top)[0]->mutable_cpu_data()[0] = 2;
  // (*top)[0]->mutable_cpu_data()[0] = 3;
  // Accuracy layer should not be used as a loss function.
} 

INSTANTIATE_CLASS(PerClassAccuracyLayer);

}  // namespace caffe
