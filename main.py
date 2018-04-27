import pandas as pd

import Consts
from Consts import RAW_FILE_PATH, RAW_SPLIT_FILE_PATH
from ElectionsDataPreperation import ElectionsDataPreperation as EDP, DataSplit
from scale_data import ScaleData
from sklearn.ensemble import RandomForestClassifier
from sfs import sfsAux
do_get_raw_data = True
do_prep_raw_data = True
do_feature_selection = True

def main():

    # FIRST STEP: Get the data and split it in to 2 groups of 3 data sets.
    # we need to bring the initial file only once. while working on it, it is rather efficient to work on local files
    # yet we'd like to be able to get the files and fall threw these steps again if needed.
    if do_get_raw_data:
        ds = DataSplit(RAW_FILE_PATH)
        ds.saveDataSetsToCsv()

    # SECOND STEP: Prepare the data for work.
    secondStepPrep_dict = dict()
    scaleData_dict = dict()
    if do_prep_raw_data:
        for i in range(1,3):
            # start the preparing data class
            secondStepPrep_dict[i] = EDP(RAW_SPLIT_FILE_PATH.format(i, "X_train", i),
                                         RAW_SPLIT_FILE_PATH.format(i, "X_val", i),
                                         RAW_SPLIT_FILE_PATH.format(i, "X_test", i),
                                         RAW_SPLIT_FILE_PATH.format(i, "Y_train", i),
                                         RAW_SPLIT_FILE_PATH.format(i, "Y_val", i),
                                         RAW_SPLIT_FILE_PATH.format(i, "Y_test", i))
            # Load the data from csv.
            # Swap strings to numeric values
            # Impute missing data
            # Impute outlier and typos
            secondStepPrep_dict[i].loadAndImpute(Consts.listAdditionalDataPreparation)

            # scale the data
            scaleData_dict[i] = ScaleData()  # type: ScaleData
            scaleData_dict[i].scale_train(secondStepPrep_dict[i].trainData)
            scaleData_dict[i].scale_test(secondStepPrep_dict[i].valData)
            scaleData_dict[i].scale_test(secondStepPrep_dict[i].testData)



    if do_feature_selection:
        # relief + sfs + correlation matrix
        for i in range(1, 3):
            # Remove features with a very high correlation
            secondStepPrep_dict[i].removeAbove95Corr()
            # create a random forest for the sfs
            rClf = RandomForestClassifier()
            max_amount_of_features = 23
            bestFeatures = sfsAux(rClf, secondStepPrep_dict[i].trainData, secondStepPrep_dict[i].trainLabels,
                                  max_amount_of_features)
            secondStepPrep_dict[i].trainData = secondStepPrep_dict[i].trainData[bestFeatures]
            secondStepPrep_dict[i].valData = secondStepPrep_dict[i].valData[bestFeatures]
            secondStepPrep_dict[i].testData = secondStepPrep_dict[i].testData[bestFeatures]
            pd.DataFrame(bestFeatures).to_csv(RAW_SPLIT_FILE_PATH.format(i,"Best_chosen_features",i))
            secondStepPrep_dict[i].trainData.to_csv(RAW_SPLIT_FILE_PATH.format(i,"Best_chosen_train",i))
            secondStepPrep_dict[i].valData.to_csv(RAW_SPLIT_FILE_PATH.format(i,"Best_val_chosen",i))
            secondStepPrep_dict[i].testData.to_csv(RAW_SPLIT_FILE_PATH.format(i,"Best_test_chosen",i))



if __name__ == "main":
    main()