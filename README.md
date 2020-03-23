# Email Automation
 Extraction of project related information from emails
The file email_automation.py is to be run, There are suggestions and instructions within this file itself to be followed.
There is a pickle file uploaded to be used as a tokenizer.
There is a weights file to be loaded to the model used in the code.(Since it's bigger than 25 mb ,I could'nt upload it)
The email automation aims at extarcting , the project number, project name, organization name, location of the project , date,time and email-ids mentioned in the email.
The used model is of deeplearning framework and uses a combination of CNN and LSTM for extraction of project numbers and for the rest fof the project related information deeppavlov module from which NER is used the pretrained model from this is loaded which is used to predict the orgainisation, country,street,locality, date ,time ,work of art etc from the content of the email
For the  content of the mail a module named mail-parser is used  to extract the relevant part of the mail from it
The nltk library is used for pre-processing of the text extracted from the mail and then tokenize it for the model for further processing.




