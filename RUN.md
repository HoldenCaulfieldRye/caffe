
1. train redbox positives
   -> scraping
      -> does zones help scraping?
	 -> yes if: num cases [¬zone & scrape] == 0
	    NO! cf sales_22-10-2014.txt
      -> added 26,354 redbox positives
         and 2485 redbox others to avoid overfit on redbox img
      -> tail and not perfect ones to reduce mislab chances
      -> threw it in Bluebox/raw_data/dump, deleted jpgs in CorrRedbox
         -> copied to graphic08 too

NEXT:	 
   -> oh no! overfitting on redbox img style
      -> remove Blueboxes from val
      -> append test.txt to train.txt
      -> change paths to Redbox when applicable
      -> rm Redboxes from data2/*/Bluebox/*
         -> ONLY DONE ON GRAPHIC05!
      -> add 20k non-scraping and non-noScrapeZone
      -> merge unsuitable photo with no scraping
         -> ./setup.py --task=scrape --box=blue --learn=6-14
      -> merge scrape zones with no scraping?
      
   -> oh no! still overfitting on redbox img style
      -> obviously, still ton more redbox positives in train
      -> bring in redbox positives and negatives, both in same amount
         as there are of bluebox positives
	 -> ./setup.py --task=scrape --box=blue --learn=6-14

2. train:
   -> joint misaligned
   -> inadequate clamping
   -> 

3. rebalanced gradient
     

4. data augmentation: rotations
   -> easy 1st step: vertical mirror

5. ROC curve for optimal threshold
   -> modify run_classifier
	 

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
   


