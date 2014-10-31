NEXT:

   -> inadcl
      -> takes 600 iters to get off the ground!!

   -> misal
      -> 0 trainerr in 20 iter, 0.5 PCA, total imbalance stuck
      -> try again:
         -> fc7 only
	 -> last time only 4 pos in val! make 15
	 -> imbalance: max redbox st 80% class, 50% blur	 
	 
2. rebalanced gradient
   -> finetune from original
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




