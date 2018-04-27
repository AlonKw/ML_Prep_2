from Consts import RAW_FILE_PATH, RAW_SPLIT_FILE_PATH
from ElectionsDataPreperation import ElectionsDataPreperation as EDP, DataSplit

get_raw_data = False

def main():

    # FIRST STEP: Get the data and split it in to 2 groups of 3 data sets.
    # we need to bring the initial file only once. while working on it, it is rather efficient to work on local files
    # yet we'd like to be able to get the files and fall threw these steps again if needed.
    if get_raw_data:
        ds = DataSplit(RAW_FILE_PATH)
        ds.saveDataSetsToCsv()

    # SECOND STEP: Prepare the data for work.
    secondStepPrep_dict = dict()
    for i in range(1,3):
        # start the preparing data class
        secondStepPrep_dict[i] = EDP(RAW_SPLIT_FILE_PATH.format(i, "X_train", i),
                                     RAW_SPLIT_FILE_PATH.format(i, "X_val", i),
                                     RAW_SPLIT_FILE_PATH.format(i, "X_test", i) )

        secondStepPrep_dict[i].load()
        # secondStepPrep_dict[i].removeAbove95Corr()

if __name__ == "main":
    main()