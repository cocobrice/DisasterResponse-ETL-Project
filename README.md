# DisasterResponse-ETL-Project

As part of a project for Udacity Data Science nanodegree I have built a web application to categorise messages based on their disaster relief context. The data was provided by 


Contents:
1. [Libraries used](#libraries-used)
2. [Data](#data)
3. [Modules](#modules)
4. [Instructions](#instructions)

### **Libraries used**
* numpy
* pandas
* scikit-learn
* json
* plotly
* sqlalchemy
* nltk
* flask
* pickle
* re
* sys

### **Data**
|      Data       |             Description                      |                    Source                      |
|-----------------|----------------------------------------------|------------------------------------------------|
|   messages.csv  |   disaster response messages   | data/messages.csv |
|   categories.csv   |   categorisation of disaster response messages                | data/categories.csv |

### **Modules**
|   Module      | Description |
|---|---|
| run.py | This notebook is used for cleaning text, finding commonly used words and phrases, using TextBlob to establish sentiment, cleaning feature data and modelling results |

### **Instructions**:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
