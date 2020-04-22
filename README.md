# law school predictor

- A web application created using AWS Elastic Beanstalk and Streamlit that allows users to input their statistics and receive a multiclass prediction from an XGBoost model using an up-sampling method to account for most recent years
- Updated weekly using cron on a python script that scrapes and parses HTML with BeautifulSoup and stores data on S3
- Validated thoroughly and most importantly passed the sniff test by current law school applicants