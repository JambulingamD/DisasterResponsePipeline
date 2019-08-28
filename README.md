# Disaster Response Pipeline Project

This project uses ETL to load the data from files into DataFrame and then into sqlite database. It then 
reads the data from sqlite database and uses ML pipeline based on ntlk to process the text data. The results 
are thereafter distributed using a web page created using flask.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
    If you get the error:
    "/opt/conda/lib/python3.6/runpy.py:125: RuntimeWarning: 'nltk.downloader' found in sys.modules after import of package 'nltk', but prior to 	execution of 'nltk.downloader'; this may result in unpredictable behaviour
    warn(RuntimeWarning(msg))"

	Run the command
    `python download_ntlk.py`
    
3. Go to http://0.0.0.0:3001/

### Files used:
`data/process_data.py`: for doing the ETL

`models/train_classifier.py`: for doing the ML Pipeline

`run.py`: to create the web page and run it in the web server using flask

`app/templates/`: web content