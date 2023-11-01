#CREATE A CHATBOT USING PYTHON

#DESCRIPTION
This project is designed to build a chatbot whose main functionality is chit-chat. The Chatbot construction involves various natural language processing techniques. The Trained model is later incorprated with user interface which is built using flask integration. The friendly user interface is designed with HTML and CSS.Python is mainly used for model development and implemetation.

## Installation & Setup

[Install Python] https://www.python.org/downloads/

[Install Anaconda] https://www.anaconda.com/download

If you have Python and anaconda installed then check their version in the terminal or command line tools

```
python3 --version

```

```
conda --version

```

[Install Visual studio] https://code.visualstudio.com/download
     Launch visual studio through Anaconda Navigator

## Installing Flask

In your terminal run  

```
conda install flask

```
## Installing torch

In your terminal run  

```
conda install torch

```
## Installing transformers

In your terminal run  

```
conda install transformers

```

## Installing spacy

In your terminal run  

```
conda install spacy

```

## What is created

 We're going to walk through the process of building a web application that can chat with users using Microsoft DialoGPT, a pre-trained language model. To create this application, we'll integrate DialoGPT with Flask, a popular Python web framework.

For the front-end of our application, we'll use HTML, CSS, and JavaScript to design an interactive chat interface. 

Throughout this, I'll provide detailed steps on setting up your development environment, installing necessary dependencies, and creating the required files and code for the chatbot application. Additionally, it's explained how to train and fine-tune the DialoGPT model to enhance the quality of its responses.

By the end , you'll have a fully functional chatbot that can engage in conversations with users.

# Dataset from kaggle
[Install Dataset] https://www.kaggle.com/datasets/grafstor/simple-dialogs-for-chatbot/download?datasetVersionNumber=2

# ChatBot Link
The Chatbot is constructed using the Microsoft/DialoGPT-medium model and customized model using given dataset.

```
https://huggingface.co/microsoft/DialoGPT-medium
```

# User-Html

```
var userHtml = '<div class="d-flex justify-content-end mb-4"><div class="msg_cotainer_send">' + user_input + '<span class="msg_time_send">'+ time + 
    '</span></div><div class="img_cont_msg"><img src="https://i.ibb.co/d5b84Xw/Untitled-design.png" class="rounded-circle user_img_msg"></div></div>';
```

# Bot-HTML

```
var botHtml = '<div class="d-flex justify-content-start mb-4"><div class="img_cont_msg"><img src="https://i.ibb.co/fSNP7Rz/icons8-chatgpt-512.png" class="rounded-circle user_img_msg"></div><div class="msg_cotainer">' + bot_response + '<span class="msg_time">' + time + '</span></div></div>';
```

