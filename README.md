# PD_Motor_Outcomes_post_DBS
 
Publishing the whole codebase for "Radiomic features of amygdala nuclei and hippocampus subfields help to predict deep brain stimulation motor outcomes for Parkinson‘s disease patients" paper.

NOTE: The codebase is extremely messy due to rapid solo development. Some minor data preparation tasks were done by hand (will be outlined in the relevant parts).

The folder "data_preparation" contains the code used for preparing the data (transforming feature matricies, joining different files together, etc.). The script inside the folder ("data_preparation.py") won't work because some of the required files were deleted (due to patient confidencielity concerns). The final data file used for further analysis generated by this script is called "Final_Data_index.csv"

The folder data_analysis contains all the files and scripts that produced the results in the paper. The scripts should be ran in this order:
1. "pearson_ftest_anova_MRMR.py". This script produces the features used in the paper into the file called "MRMR2.csv" using the MRMR feature selection algorightm (Pearson correlation for redundancy and one-way ANOVA F-test for relevance). This file is then transformed into "MRMR_2.csv" by hand changing the labels from the encoded shortened form to original. For this step there is another MRMR feature selection script "spearman_kruskall_wallis_MRMR.py" which selects features using the MRMR algorigthm with Spearman corelation for redundancy and Kruskal–Wallis test for relevance. Both scripts produce features from the left hemisphere although the features themselves are different. To reiterate features used in the paper are produced by the script "pearson_ftest_anova_MRMR.py".
2. "model_classification.py" performs model training and predictions for different models and saves the results into files with a ".npy" ending. It performs these actions using the file "MRMR2.csv" containing the selected features.
3. "draw_ROCs.py", "draw_correlation_heatmap.py" and "Graphing/draw_boxplots.py" produces the figures seen in the paper in a ".tif" format.