# Heart-Disease-Diagnosis
Flask based web app to diagnose the patient using Python3, Predicts the presence of one of four types of heart disease(or none at all) using a patient's medical test report data.

## Dataset
The [Heart disease data set](https://archive.ics.uci.edu/ml/datasets/heart+Disease) consists of patient data from Cleveland, Hungary, Long Beach and Switzerland. The combined dataset consists of 14 features and 916 samples with many missing values. 
The features used in here are,
1. age: The patients age in years
2. sex: The patients gender(1=male; 0=female)
3. cp: Chest pain type,
	*Value 1: typical angina 
	*Value 2: atypical angina 
	*Value 3: non-anginal pain 
	*Value 4: asymptomatic 
4. trestbps: Resting blood pressure (in mm Hg on admission to the hospital)
5. chol: Serum cholestoral in mg/dl
6. fbs: Fasting blood sugar > 120 mg/dl? (1=true; 0=false) 
7. restecg: Resting electrocardiographic results
	*Value 0: normal 
	*Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) 
	*Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria 
8. thalach: Maximum heart rate achieved
9. exang: Chest pain(angina) after exercise? (1=yes; 0=no)
10. thal: Not described 
	*Value 3=normal
	*Value 6=treated defect 
	*Value 7=reversible defect 
11. num: Target
	*Value 0: less than 50% narrowing of coronary arteries(no heart disease)
	*Value 1,2,3,4: >50% narrowing. The value indicates the stage of heart disease

Dataset creators,
1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D. 
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D. 
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D. 
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D. 

## Running the web app
#### Locally
- Install requirements  
   `pip install -r requirements.txt`
- Run flask web app  
    `python main_file.py`

## Models used and accuracy
A Random forest classifier achieves an average multi-class classification accuracy of 56-60%(183 test samples).
It gets 75-80% average binary classification accuracy(heart disease or no heart disease).
