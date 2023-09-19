from langchain.llms import OpenAI, OpenAIChat
from langchain import PromptTemplate, LLMChain
from collections import OrderedDict
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from langchain.chat_models import ChatOpenAI


class Parse(BaseModel):
    code: str = Field(description="only provide a runnable python code as per the user's instruction")
    explain_code: str = Field(description="explanation of the generated code")

global ops_db

ops_db = OrderedDict()

def initialize_llm(external_variable):
    external_variable = "some_value"
    return external_variable
    
def llm(prompt):
  return chat.predict(prompt)


def vars_features(**vars):

  k = 1

  vars_features={}

  for var_name, var_value in vars.items():

    features = {}

    features['data_type'] = str(type(var_value))

    features['memory_location'] = id(var_value)

    questions = f"""
    Answer the following questions about this {type(var_value)} data type:
    1) does this {type(var_value)} data type have columns?
    2) does this {type(var_value)} data type have columns names or labels?
    3) does this {type(var_value)} data type have rows?
    4) does this {type(var_value)} data type have rows names or index labels?

    write a python function called "cols_and_rows()" that takes no input but returns the
    answers of the previous questions as a python list such as ['yes', 'no', 'yes', 'no'].
    you should only provide runnable python code without further comments or anything else.
    """
    x = llm(questions)
    exec(x, globals())
    answers = cols_and_rows()
    features['columns_available'] = answers[0]
    features['columns_names_available'] = answers[1]
    features['rows_available'] = answers[2]
    features['rows_names_available'] = answers[3]

    if features['columns_names_available'] == 'yes':
      code_cols = llm(f"""
      write a python function called return_column_names that takes a variable X
      with {type(var_value)} data type and returns that a list of columns names. provide only runnable python code
      without further comments or anything else.
            """)
      exec(code_cols, globals())
      features['columns_names'] = return_column_names(var_value)
    else:
      features['columns_names'] = None

    dim_code = llm(f"""
    write a python function called var_dim that takes a variable X
    with {type(var_value)} data type and returns the number of dimensions of it. provide only
    runnable python code without further comments or anything else.
    """)
    exec(dim_code, globals())
    features['dimension'] = var_dim(var_value)

    shape_code = llm(f"""
    write a python function called var_shape that takes a variable X
    with {type(var_value)} data type and returns the shape of it. provide only
    runnable python code without further comments or anything else. if this data type does not have
    shape attribute, then then var_shape function must return None instead. for example, if the data type
    is a pandas dataframe, then the function will look like this:

    def var_shape(X):
      return X.shape

    the var_shape(X) was written in this way becuase pandas dataframe type has .shape method to check for shape.

    However, if the data type is an integer for example, then the function will look like this:
    def var_shape(X):
      return None

    the var_shape(X) was written in this way becuase integers in python do not have shape
          """)
    exec(shape_code, globals())
    features['shape'] = var_shape(var_value)

    vars_features[var_name] = features

    print(f"{var_name} variable features extracted ({str(k)}/{str(len(vars))})")
    k+=1
  print('--> All variables features were extracted.\n')
  return vars_features


def vars_features_to_string(vars_features):
  s=''
  for var_name in vars_features.keys():
    s+= f"\n* the features of python variable '{var_name}' are:\n"
    features = vars_features[var_name]
    for feature in features.keys():
      s+= f"  {feature} of {var_name} is {features[feature]}\n"
  return s



def inspect_and_execute(code_to_execute):
  initial_globals = globals().copy()
  exec(code_to_execute, globals())
  new_variables = {
      name: value
      for name, value in globals().items()
      if name not in initial_globals or initial_globals[name] is not value
  }
  return new_variables



def instruct_llm(command, run_code = True, vars_description = "", **vars):
  input_variables_dict = {"query":command}
  # Set up a parser + inject instructions into the prompt template.
  parser = PydanticOutputParser(pydantic_object=Parse)
  instruction_template = "Answer the user query.\n{query}\n"
  if len(vars) > 0:
    instruction_template += "{vars_features_template}\n"
    vars_features_str, input_variables_dict["vars_features_template"] = vars_features_template_gen(**vars)
  if len(vars_description) > 0:
    instruction_template += "{vars_description_template}\n"
    input_variables_dict["vars_description_template"] = vars_description_template(vars_description)
  instruction_template += "\n{format_instructions}"

  prompt = PromptTemplate(
      template = instruction_template,
      input_variables = list(input_variables_dict.keys()),
      partial_variables={"format_instructions": parser.get_format_instructions()},
  )
  _input = prompt.format_prompt(**input_variables_dict)
  output = llm(_input.to_string())
  output_parse = parser.parse(output)
  code = output_parse.code.replace('```python','').replace('```','')
  #code = output_parse.code.replace('```python','').replace('```','').replace('fare','Fares')
  explain_code = output_parse.explain_code
  database={}
  if len(vars) > 0:
    database['vars_features'] = vars_features_str
    database['given_vars'] = list(vars)
  database['instruction_template'] = instruction_template
  database['generated_code'] = code
  database['explain_code'] = explain_code
  database['run_code'] = run_code
  database['llm_input'] = _input
  error_message = None
  if run_code == True:
    try:
      exec_vars = inspect_and_execute(code)
      database['variables'] = exec_vars
      print("...............\nThe LLM's generated code was executed as per the command, and the following variables were created within the execution function:",exec_vars.keys())
    except Exception as e:
      error_message = str(e)
      database['error_message'] = error_message
      print("\nError message after executing the generated code.\nHere is the error message:\n", error_message,'\n')
  ops_db[command] = database
  return code, explain_code, error_message



def vars_features_template_gen(**vars):

  vf = vars_features(**vars)
  vars_features_str = vars_features_to_string(vf)
  return vars_features_str, f"""\n
The user's instruction deals with this list of variables {list(vars)} I will help you understand those variables through their features stored in
python dictionary. This dictionary is essentially a mapping of variable names to their corresponding properties. Each key in the dictionary is a string
representing the variable name, and each value is another dictionary representing the properties of the corresponding variable. Here's a brief explanation
of each property:

1) data_type: The type of the data that the variable is storing. This could be any Python data type, but in your examples, it includes types like
              pandas.core.frame.DataFrame, numpy.ndarray, torch.Tensor, and float.
2) memory_location: The memory location of the variable, which is essentially a unique identifier for the location in memory where the data associated
                      that could be thought of as having columns, but a single float value does not.

4) columns_names_available: This indicates whether the data type of the variable has named columns. For example, a Pandas DataFrame has named columns, but a Numpy
                            array or a Torch tensor does not.

5) rows_available: Similar to columns_available, but for 'rows'.

6) rows_names_available: Similar to columns_names_available, but for row indices. For example, pandas DataFrames support named rows (which are called
                          indices in pandas).

7) columns_names: If the variable's data type supports named columns, this entry lists those names. Otherwise, it is None.

8) rows_indices: If the variable's data type supports indexing its rows, this entry lists those indices. Otherwise, it is None.

9) dimension: The number of dimensions that the variable's data has. For example, a 2D array like a Pandas DataFrame or a Numpy array has 2 dimensions.
              A single float value is 0-dimensional.

10) shape: The shape of the data in the variable. This is typically a tuple where the nth entry indicates the size of the data along the nth dimension. For example,
          a DataFrame with 3 rows and 2 columns would have a shape of (3, 2). A single float value, being 0-dimensional, has no shape and hence it is None.

In summary, this dictionary provides an extensive mapping of variable names to the properties of the data those variables are storing. This could be useful in many
contexts, such as data analysis, debugging, or any situation where you need to keep track of a large number of variables and their properties. Take those features of
the variables into account when performing the user's instruction. Here are the features for each of the variables that will be used in the code generated for the user:{vars_features_str}"""

def vars_description_template(vars_description):
  return f"""
Now that you know the features of the user's variables, you need to understand the description or meaning for each variable. The reason why I need you to
understand the description for each variable is that it helps you:

1) Code Understanding: Understanding variable names and their types helps the model understand the logic and structure of the code. For instance, knowing that a
  variable is a DataFrame type helps the model predict that DataFrame operations might be applied to it.

2) Intelligent Suggestions: If the model knows the type of a variable, it can provide more accurate suggestions or code completions when you're writing code.
  For example, if the model knows a variable is a string, it can suggest string methods when you're trying to perform operations on that variable.

3) Error Prevention: If the model understands variable types and uses, it can help you avoid common errors, like type mismatches or using a method that doesn't
  apply to a given data type.

4) Better Explanations: When answering questions or explaining code, understanding the variables in a program allows the model to provide more precise and contextual
  explanations. For instance, knowing that a variable is used as a counter in a loop can help the model explain the loop's purpose.

5) Code Refactoring: Knowing the roles and types of different variables can help suggest ways to refactor or optimize the code.

6) Appropriate Variable Use: Understanding the purpose and type of a variable allows a language model to use that variable correctly when generating code.
  For instance, if a variable is a list of items, the language model will know that it can perform operations like iterating over the list or adding elements
  to it.
7) Context-Aware Code: By understanding the variables, the model becomes aware of the code's context and can generate code that fits that context. For example,
  if a variable is defined as a database connection, the model will generate code relevant to database operations.

8) Code Optimization: With knowledge of the variables, their types and uses, the model can generate optimized code. It can suggest more efficient data structures
  or algorithms based on the variable types.

9) Improved Code Readability: Understanding the variable's role helps the model name new variables more appropriately, leading to more readable and maintainable code.

In summary, understanding variables and their usage is a key aspect of code comprehension and provides a foundation for many useful code-related tasks and features
that can be supported by large language models. Take the description of the variables into account when performing the user's instruction. Here is the description for
each of the variables: \n{vars_description}.
"""

def fix_code_template(from_command, source_code):
  instruction_template_without_format = ops_db[from_command]['llm_input'].text.split("The output should be formatted as a JSON instance that conforms to the JSON schema below.")[0]
  s = f"""This is the erroneous code:\n{ops_db[from_command][source_code]}\nThe generated error message from the erroneous code above is:\n'{ops_db[from_command]['error_message']}'\n
  Here is the instruction template that was used to generate the erroneuos code by the large langauge model (LLM):\n{instruction_template_without_format}\n
      """
  return s

def error_handling(from_command = None, custom_fix_template = None, run_code = True, source_code = 'generated_code', custom_input_variables={}):

  parser = PydanticOutputParser(pydantic_object=Parse)
  fix_template = """{query}:\n"""
  if from_command is not None:
    fix_template += fix_code_template(from_command, source_code)
  else:
    fix_template += custom_fix_template
  fix_template += "\n{format_instructions}"
  if len(custom_input_variables) == 0:
    in_vars = ['query']
  else:
    in_vars = list(custom_input_variables.keys())
  prompt = PromptTemplate(
      template = fix_template,
      input_variables = in_vars,
      partial_variables={"format_instructions": parser.get_format_instructions()},
  )
  if len(custom_input_variables) > 0:
    _input = prompt.format_prompt(**custom_input_variables)
  else:
    _input = prompt.format_prompt(query="fix this code:\n")
  ops_db[from_command]['llm_input_fix'] = _input.text
  output = llm(_input.to_string())
  output_parse = parser.parse(output)
  fixed_code = output_parse.code.replace('```python','').replace('```','')
  explain_fixed_code = output_parse.explain_code
  error_message = None
  if run_code == True:
    try:
      exec_vars = inspect_and_execute(fixed_code)
      if from_command is not None:
        ops_db[from_command]['variables'] = exec_vars
        ops_db[from_command]['fixed_code'] = fixed_code
      print("...............\n--> The LLM's generated code was fixed and executed as per the command, and the following variables were created within the execution function:",exec_vars.keys())
    except Exception as e:
      error_message = str(e)
      ops_db[from_command]['fixed_code'] = fixed_code
      ops_db[from_command]['error_message_after_fix'] = error_message
      print("\nThere was an error even after tryubg to fix the code! Here is the error message:\n", error_message)
  return fixed_code, explain_fixed_code, error_message


def auto_handle(command, vars_description = "", max_fixes = 3, run_code = True, **vars):

  k=1
  code, explain_code, error_message = instruct_llm(command, run_code = run_code, vars_description = vars_description, **vars)
  original_code = code
  while (k <= max_fixes) and (error_message is not None):
    print(f'--> This is the {k}/{max_fixes} try to fix the erroneous code:')
    error_message = None
    if k == 1:
      source_code = 'generated_code'
    else:
      source_code = 'fixed_code'
    code, explain_code, error_message = error_handling(from_command = command, run_code = run_code, source_code = source_code)
    if error_message is not None:
      print('The erroneous code was still not fixed. The new error message is:\n',error_message)
    else:
      print('\n--> Here is a comparison between the first generated code and the fixed one:')
      print(llm(f"""This is the original code generated which which outputs an error:\n{original_code}\nHere is the fixed code:\n{code}\nExpalain the difference between the original code generated and the fixed one.""").replace(". ",".\n"))
    k += 1
  else:
    print(f'--> Number of code fixes attempted k = {k-1}.')
    if error_message is not None:
      print('The original code generated by the LLM was unfortunately not fixed, so try to manually adjust it')
  return code, explain_code, error_message
