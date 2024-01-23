import os

def run_data_pull():
    os.system('python Stats_Data_Pull.py')
    os.system('python Pull_Todays_Games.py')

def run_data_preparation():
    os.system('python Data_Prep.py')

def run_model_training():
    os.system('python Model_Training.py')

def run_model_prediction():
    os.system('python Evaluate_Model.py')

def main():
    print("Running Data Pull...")
    run_data_pull()

    print("Running Data Preparation...")
    run_data_preparation()

    print("Running Model Training...")
    run_model_training()

    print("Running Model Prediction...")
    run_model_prediction()

    print("Pipeline execution completed.")

if __name__ == "__main__":
    main()
