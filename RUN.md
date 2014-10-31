NEXT:

   UPDATE flag_kookup.txt BECAUSE UNSUITABLE PHOTO MISSING!
   -> then train with new additional data and rebalanced grad

   -> misal
      -> 0 trainerr in 20 iter, 0.5 PCA, total imbalance stuck
      -> try again:
         -> fc7 only
	 -> last time only 4 pos in val! make 15
	 -> imbalance: max redbox st 80% class, 50% blur	 
	 
   -> inadcl 06
      -> 0 trainerr in 20 iter, 0.5 PCA, total imbalance stuck
      -> try again: 97, 52, 74, 88
         -> fc7 only
	 -> imbalance: max redbox st 50% class, 70% blur
	 -> maybe inadcl semantically hard though

   -> water 09
      -> 0 trainerr in 20 iter, maybe 0.6 PCA but looks like noise
      -> try again: 79, 77, 78, 78
         -> fc7 only
	 -> imbalance: max redbox st no imbalance anywhere
      -> improve:
         -> max redbox 78% class imbalance
	 -> fc7 then fc6?

2. rebalanced gradient
   -> 

3. data augmentation: rotations
   -> easy 1st step: vertical mirror
   -> ON GRAPHIC07

4. ROC curve for optimal threshold
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




