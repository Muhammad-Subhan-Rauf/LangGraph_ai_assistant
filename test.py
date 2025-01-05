import subprocess
import google.generativeai as genai
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import Graph, END
import os
import glob

# Configure Gemini API Key
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyDvbke4TODM1nOMbkZAXXhOVGQeECSsATU")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# Define State
class AgentState:
    def __init__(self, user_input=None, is_command=False, shell_command=None, response=None, execution_result=None, error=None, memory=None):
        self.user_input = user_input
        self.is_command = is_command
        self.shell_command = shell_command
        self.response = response
        self.execution_result = execution_result
        self.error = error
        self.memory = memory

# Step 1: Determine if Input is a Command
def classify_input(state: AgentState) -> AgentState:
    print(f"Input received: {state.user_input}")
    prompt_template = PromptTemplate(
        input_variables=["input"],
        template="""
        Classify the following user input as either a 'command' or 'chat':
        Input: {input}
        Output (only 'command' or 'chat'):
        """
    )
    chain = prompt_template | llm
    result = chain.invoke({"input": state.user_input})
    classification = result.content.strip().lower()
    state.is_command = (classification == "command")
    print(f"Classification: {'Command' if state.is_command else 'Chat'}")
    return state

# Step 2: Process Chat Input (Updated for Memory)
def handle_chat(state: AgentState) -> AgentState:
    # Add user input to memory
    state.memory.add({"role": "user", "content": state.user_input})

    # Handle date and time requests directly
    if "date" in state.user_input.lower():
        state.response = f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."
        print(state.response)
    elif "time" in state.user_input.lower():
        state.response = f"The current time is {datetime.now().strftime('%H:%M:%S')}."
        print(state.response)
    else:
        # Use memory in the conversation prompt
        chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in state.memory.load()])
        prompt_template = PromptTemplate(
            input_variables=["history", "input"],
            template="""
            You are a helpful assistant. Use the following conversation history to provide a coherent response:
            Conversation History:
            {history}

            User Input: {input}
            Response:
            """
        )
        chain = prompt_template | llm
        result = chain.invoke({"history": chat_history, "input": state.user_input})
        state.response = result.content.strip()
        print(f"Chat Response: {state.response}")
    
    # Add the AI response to memory
    state.memory.add({"role": "assistant", "content": state.response})
    return state

# Step 3: Interpret Command Input
def interpret_command(state: AgentState) -> AgentState:
    prompt_template = PromptTemplate(
        input_variables=["input"],
        template="""
        Convert the following user request into a shell command, resolving any file paths or folder paths using the user's context. If it includes a file or folder path, resolve the full path based on the current directory of the user. If a file path is ambiguous due to the presence of wild cards in the path, use glob to get a list of possible files and try the command once with each resolved file. If no files are found, then consider the file path is invalid. If no file or folder is used in the command or no paths are found then return the command. If a file or folder path does not exist then return the 'Unsupported Command' message. Respond with only the shell command or 'Unsupported Command'
        User Request: {input}
        Shell Command:
        """
    )
    chain = prompt_template | llm
    result = chain.invoke({"input": state.user_input})
    state.shell_command = result.content.strip()
    print(f"Generated Shell Command: {state.shell_command}")
    return state

# Step 4: Check and Install Missing Packages
def install_missing_packages(command):
    try:
        # Check if command exists
        subprocess.check_output(f"which {command.split()[0]}", shell=True, text=True)
    except subprocess.CalledProcessError:
        print(f"Dependency missing for: {command.split()[0]}")
        print(f"Installing {command.split()[0]}...")
        try:
            subprocess.run(f"brew install {command.split()[0]}", shell=True, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error installing {command.split()[0]}: {e.stderr.decode()}")
            raise Exception(f"Error installing {command.split()[0]}: {e.stderr.decode()}")
        except Exception as e:
            print(f"An unknown error occurred during installation: {e}")
            raise Exception(f"An unknown error occurred during installation: {e}")

# Step 5: Execute Shell Command
def execute_command(state: AgentState) -> AgentState:
    shell_command = state.shell_command

    # Handle unsupported commands
    if "Unsupported Command" in shell_command:
        state.execution_result = "Command not supported or cannot be executed."
        print(state.execution_result)
        return state

    # Check dependencies and install if missing
    try:
      install_missing_packages(shell_command)
    except Exception as e:
        state.execution_result = f"Error during dependency installation: {e}"
        print(state.execution_result)
        return state


    try:
        print(f"Executing Command: {shell_command}")
        result = subprocess.check_output(shell_command, shell=True, stderr=subprocess.STDOUT, text=True)
        state.execution_result = result
        print(f"Command Output:\n{result}")
    except subprocess.CalledProcessError as e:
        state.execution_result = f"Error executing command: {e.output}"
        print(state.execution_result)
    except Exception as e:
        state.execution_result = f"Execution failed: {str(e)}"
        print(state.execution_result)
    return state

# Step 6: Respond to Command Execution
def generate_command_response(state: AgentState) -> AgentState:
    prompt_template = PromptTemplate(
        input_variables=["command", "result"],
        template="Explain the following command and its output:\nCommand: {command}\nOutput: {result}"
    )
    chain = prompt_template | llm
    result = chain.invoke({"command": state.shell_command, "result": state.execution_result})
    state.response = result.content.strip()
    print(f"Explanation: {state.response}")
    return state

# Step 7: Handle Errors 
def handle_error(state: AgentState) -> AgentState:
    prompt_template = PromptTemplate(
        input_variables=["error_message"],
        template="""
        An error occurred during the execution of the program. Analyze the error message and provide a suggestion on how to resolve it, or an explanation as to why it occurred. If there was an error during dependency installation, recommend the user to retry the command.
        Error: {error_message}
        Response:
        """
    )
    chain = prompt_template | llm
    result = chain.invoke({"error_message": state.error})
    state.response = result.content.strip()
    print(f"Error Handling Response: {state.response}")
    return state

# Create LangGraph Workflow
def create_agent_graph():
    graph = Graph()
    graph.add_node("classify_input", classify_input)
    graph.add_node("handle_chat", handle_chat)
    graph.add_node("interpret_command", interpret_command)
    graph.add_node("execute_command", execute_command)
    graph.add_node("generate_command_response", generate_command_response)
    graph.add_node("handle_error", handle_error) # Added the new error handling node
    
    graph.set_entry_point("classify_input")
    
    graph.add_conditional_edges(
        "classify_input",
        lambda state: "handle_chat" if not state.is_command else "interpret_command"
    )
    
    graph.add_edge("handle_chat", END)
    graph.add_edge("interpret_command", "execute_command")
    
    # Conditional edge for execute command. If it encounters an error, route to error handling
    graph.add_conditional_edges(
      "execute_command",
        lambda state: "generate_command_response" if not state.execution_result.startswith("Error") and not state.execution_result.startswith("Execution failed") and not state.execution_result.startswith("Error during dependency installation") else "handle_error"
      )
    
    graph.add_edge("generate_command_response", END)
    graph.add_edge("handle_error", END) # Error handler routes to the end node


    return graph

# Command Line Interface (Updated for Memory)
def main():
    print("AI Command Line Agent with Memory (powered by Gemini) - Type 'exit' to quit.")
    graph = create_agent_graph()
    app = graph.compile()

    # Initialize memory and pass it to the state
    chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        # state = AgentState(user_input=user_input, memory=chat_memory)
        state = AgentState(user_input=user_input, memory=chat_memory)
        try:
            result = app.invoke(state)
            print("AI Agent:", result.response)
        except Exception as e:
            state.error = str(e)
            print("An error occurred:", state.error)
            result = app.invoke(state)
            print("AI Agent:", result.response)

if __name__ == "__main__":
    main()