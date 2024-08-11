# Abalone Age Prediction - Web App

![image](https://github.com/user-attachments/assets/efad605c-5df4-41b3-bae3-cc54e7a5605d)


## 🌟 Highlights

* Successfully deployed a machine learning web app in a user-friendly format.
* Achieved a **56% accuracy** in the prediction of Abalone tree age with linear regression.

## ℹ️ Overview

Machine learning is useful for the prediction of some desired feature in real world problems. To captre the public's attnetion, data from the pipeline must be presented in a enganging manner. Here, I used streamlit as the medium to deploy my python linear regression model on the Abalone dataset to predict the number of rings/ its age.

## 🚀 Usage

**a) Requirements**
* python 3.X (I used Python 3.10.9)
* numpy (I used numpy version 1.23.5)
* streamlit
* sklearn (version 1.5.1)

**b) Deployment**

After downloading both .py and.csv file, cd into file directory where you downloaded the two files and run in terminal of IDE (I used VSCode):

```

python -m streamlit run Prediction_Abalone.py

```
**c) Dataset + Model Analysis**

File used: Abalone_csv.ipynb

  i) The model could not process strings in the 'Gender' column of the dataframe, so a dictionary was used to convert each gender option to a number.

```

sentiment_labels_to_numbers = {
    "M" : 0,
    "F" : 1,
    "I": 2
}


```

  ii) Model performance accuracy was measured using

```

print('Accuracy of linear regression classifier on test set: {:.2f}'.format(linreg.score(X_test, y_test)))

>>> Accuracy of linear regression classifier on test set: 0.54

```


## ⬇️ Installation

type 
```
pip install {package}
```

**packages:**

* streamlit
* pandas
* numpy
* sklearn


## 💭 Future Improvements

* Round number of rings to nearest whole number for "Predicted Rings"
* Change Input Parameter scale to alllow prediction of larger values
* include the original dataset with original genders then used preprocesssed csv for machine learning step

