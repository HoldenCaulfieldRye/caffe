
backtrack what needs to be modified
===================================
solver.cpp
----------
test_score.size()
test_score.push_back(result_vec[k])
result_vec = result[j]->cpu_data()
result = test_net->Forward(bottom_vec, &iter_loss)

net.cpp
-------
Forward: return ForwardPrefilled(loss)
ForwardPrefilled: return net_output_blobs_
-- either:
Forward: net_output_blobs_[i]->ToProto(blob_proto_vec.add_blobs())
Forward: BlobProtoVector blob_proto_vec
-- or:
ForwardPrefilled: ForwardFromTo(0, layers_.size() - 1)


MODIFY: 
Net::Forward(bottom_vec, &iter_loss)




NEXT STEPS:
1) run on redbox
   1000, 0.1771
   2000, 0.1761
2) research
   -> bring in bayesian cross entropy etc
   -> confusion matrix to see evolution over training
   -> caffe mnist and imagenet examples to quickly test things out
   and use undersampling as well
3) logistic regression confidence intervals
   -> bit.ly/1oawXcH
   -> bit.ly/1oawVkS
   ->
4) python layers for caffe: bit.ly/1Dhl8Ex
5) more data augmentation
6) final product merge test set into training set!!
   



