# PyGenX

The goal of this repo is to teach you how to build a tool that helps increasing programmers productivity by utilizing LLMs for zero-code development. This tool is called PyGenX which stands for Python Generator and Executor. This tool basically generates Python code and executes it based on a user’s prompt or command. Now, there has been increasing concerns about data privacy and security when using AI models like ChatGPT for various purposes including data analysis, machine learning and and other data related tasks. Concerns often revolve around data storage and retention, data usage, data sharing with third-party, and other ethical considerations. With such concerns in mind, PyGenX can take care of data privacy when zero-code developing in Python. This is done through data-agnostic techniques which are designed to perform the same programmers’ tasks without accessing, storing, processing, or utilizing user-specific data. In this way, LLMs vendors are not exposed to your data even if those LLMs run in the cloud such as GPT-4 or GPT-3.5-turbo. Running PyGenX locally—i.e., on your own hardware—offers several advantages, especially if you’re concerned about privacy, latency, and data control. Here are the main benefits as a result of using PyGenX:

* Data Privacy: Running the application locally ensures that your data never leaves your machine, offering a high level of data security and privacy.
* Low Latency: Local execution can eliminate the round-trip time to the cloud server, resulting in faster response times. This is important for real-time applications.
* Customizations: Running locally can offer greater flexibility in terms of customization and integration with other local systems and data sources.
* Resource Utilization: When you run the LLM-generated code on your local machine, you have more control over hardware resource utilization of CPU, GPU and memory.

# The Architecture of PyGenX

The PyGenX library for zero-code development in Python requires three main inputs to it:

* User’s command: this is where the user inputs a command or prompt stating the programming task. This is used by the LLM to generate a Python code that performs the required task. 
* User’s variables: This is an optional input argument where the user can inject the variables related to his/her command for further processing. Variables could be anything from integers values to full datasets. PyGenX is data-agnostic which means it abstractly deals with the features of data not the data itself. This is important in order to preserve data privacy when using cloud based LLM such as OpenAI models. The LLM understands the related variables through its features without looking at the datapoints of it. Variables features will be discussed later on. 
* Variable’s description: this is an optional parameter where the user can explain the various aspects about the user’s variables. For example, variables description of a Pandas dataset could be the explanation and definition of each column in the dataset. 

each of the inputs discussed are converted to prompt templates before being injected to the LLM. Those prompt templates will be discussed later on as well. The PyGenX library has five main functionalities:

* Generate Code and Comments: this is where PyGenX generates the code as per the prompt in the user’s command and explains the generated code as well. The output of this block is structured using a parser from LangChain in order to separate the generated code from any other comments so that the generated code can be easily extracted from the output of the LLM and executed.
* Error Handling: LLM are not perfect, so this block is used to fix the generated code if it throws an error. Similarly, this block also uses a parser from LangChain in order to separate the generated code from any other comments.
* Auto Code Handling: this block combines both Generated Code and Comments and Error Handling in order to automatically fix any generated code. 
* Inspect and Execute Code: in this block, the user can choose whether or not the generated Python code by the LLM is executed. The inspection part of this block is see the newly instantiated variables within the generated and executed code.  
* Operations Database: this database preserves everything related to history of the user’s command such as: all created prompt templates, error messages, user’s variables, variable features, code execution and LLM outputs. This is especially important when creating the code fix prompt template that is passed to the Error Handling block when the generated code throws an error. 

PyGenX greatly increases programmers productivity for various application such as: data analysis and visualization, machine learning and deep learning development, code documentation and refactoring, automation and scripting, file operations, web development, and other software development applications. Basically, the strength and effectiveness of zero-code programming in Python using PyGenX depends on the power of the LLM used with it. Here are some examples:

## Example 1:

In this example, we are using the auto_handle() function to ask the LLM to generate and execute this user command "get the sum of fares in titanic dataset. Print and explain your results". PyGenX takes user prompt as input and returns Python objects in return. Notice that a description of the dataset was passed to the function along with a variable containing the dataset. It is important to pass the dataset to the auto_handle() function so that the dataset is processed by the generated Python code.

![alt text](https://github.com/yahya-bader-khawam/PyGenX/blob/4e2736b49d034651402eb6e615885918e290598c/Screenshot%202023-09-18%20at%203.32.15%20PM.png)

## Example 2:

In this example, PyGenX is prompted with the statement "create two random matricies in numpy and make a dot product between them. print and explain your results.". As you can see in the output, PyGenX returns a array as a result of multiplying two random matrices.

![alt text](https://github.com/yahya-bader-khawam/PyGenX/blob/17a4965dda2f5858c3f3938660444bb5f8623dcd/Screenshot%202023-09-18%20at%203.32.50%20PM.png)

## Example 3:

In this example, PyGenX is prompted with the statement on how many people survided as per titanic dataset. The dataset and its description were passed to the auto_handle() function.

![alt text](https://github.com/yahya-bader-khawam/PyGenX/blob/17a4965dda2f5858c3f3938660444bb5f8623dcd/Screenshot%202023-09-18%20at%203.33.32%20PM.png)

## Example 4:
This is a machine learning example on using auto_handle() function from PyGenX to predict the survival status in the titanic dataset.

![alt text](https://github.com/yahya-bader-khawam/PyGenX/blob/17a4965dda2f5858c3f3938660444bb5f8623dcd/Screenshot%202023-09-18%20at%203.34.46%20PM.png)
