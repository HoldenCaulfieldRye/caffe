
1. train redbox positives
   -> scrape tasks
      -> does zones help scraping?
         -> yes if: ¬[¬zone & scrape]
	 -> yes if: num cases [¬zone & scrape] == 0
	    NO!
      -> include other Redbox as well so no overfit
         on Redbox for flag detection
	 -> throw in 2485 others
	    tail and not perfect ones to reduce mislab chances
      -> throw in Bluebox/raw_data/dump
      -> delete jpgs in CorrRedbox
	 

DATA ISSUES:
        scrape   ¬scrape | total
------+--------+---------|-------
zone  | 84,163   19,857  | 104,020
¬zone | 16,480    6,497  |  22,977
------+--------+---------|
total |100,643   26,354  | 126,997     

-> need:
        scrape   ¬scrape |
------+--------+---------|
zone  |                  |
¬zone |  N/A             |
------+--------+---------|




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
   


