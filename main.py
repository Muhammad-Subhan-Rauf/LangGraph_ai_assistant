import subprocess
import google.generativeai as genai
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import Graph, END, START
import ipAddress as ipComms
import os
from commands import ubuntu_commands 

present_path = "/"
# Configure Gemini API Key
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyDvbke4TODM1nOMbkZAXXhOVGQeECSsATU", verbose= False)

# Define State
class AgentState:
    def __init__(self, user_input=None, is_command=False,  shell_command=None, response=None, execution_result=None):
        self.user_input = user_input
        self.is_command = is_command

        self.shell_command = shell_command
        self.response = response
        self.execution_result = execution_result

# Step 1: Determine if Input is a Command
def classify_input(state: AgentState) -> AgentState:
    # print(f"Input received: {state.user_input}")
    prompt_template = PromptTemplate(
        input_variables=["input"],
        template="""
        Is the user asking you to run a linux command line command if yes then return "command" else return "chat":
        Input: {input}
        Output (only 'command' or 'chat'):
        """
    )
    chain = prompt_template | llm
    result = chain.invoke({"input": state.user_input})
    classification = result.content.strip().lower()
    print("CLASSS: ",classification)
    state.is_command = (classification != "chat")

    print(f"Classification: {'Command' if state.is_command else  'Chat'}")
    return state

# Step 2: Process Chat Input
def handle_chat(state: AgentState) -> AgentState:
    # Handle date and time requests directly
    llm.temperature=0.7
    if "date" in state.user_input.lower():
        state.response = f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."
        print(state.response)
        return state
    elif "time" in state.user_input.lower():
        state.response = f"The current time is {datetime.now().strftime('%H:%M:%S')}."
        print(state.response)
        return state

    # For other chats, ask Gemini
    prompt_template = PromptTemplate(
        input_variables=["input"],
        template="You are a helpful assistant. Respond to the following: {input}"
    )
    chain = prompt_template | llm
    result = chain.invoke({"input": state.user_input})
    state.response = result.content.strip()
    print(f"Chat Response: {state.response}")
    return state

# Step 3: Interpret Command Input
def interpret_command(state: AgentState) -> AgentState:
    llm.temperature=0
    prompt_template = PromptTemplate(
        input_variables=["input"],
        template="""
        Convert the following user request into a shell command, or reply 'Unsupported Command' if it cannot be executed. Do NOT use any command 
        that changes the PWD, and when executing multiple commands at once, Always use absolute paths and dont use CD and dont forget the tilde (~):
        User Request: {input}
        Shell Command:
        """
    )
    chain = prompt_template | llm
    result = chain.invoke({"input": state.user_input})
    if result.content.strip().lower() != "unsupported command":

        state.shell_command = result.content.strip()
        print(f"Generated Shell Command: {state.shell_command}")
    return state

# **NEW STEP: Check if Command is Network Related**
def check_network_command(state: AgentState) -> AgentState:
    network_commands = ["ping", "traceroute", "nslookup", "dig", "curl", "wget", "ifconfig", "ip", "ssh", "scp", "netstat"]
    if state.shell_command:
        command = state.shell_command.replace("Shell Command: ","")
        if any(cmd in command.lower().split() for cmd in network_commands):
        
            print("Network Command Detected")

            state.execution_result = ipComms.get_network_ip_addresses()[0]

    return state

# Step 4: Check and Install Missing Packages
def install_missing_packages(command):

    command = command.replace("Shell Command: ","")
    # print(f'\n\n\n\n\nCommand: {command}')
    # print(f'\n\n\n\n\nCommand: {command.split()[0]}')
    try:
        command = command.replace("Shell Command: ","")
        # print(f'\n\n\n\n\nCommand: {command}')
        # print(f'\n\n\n\n\nCommand first: {command.split()[0]}')
        command_lst = command.split()
        # print(f"\n\n\ncommand list: {command_lst}")
        # Check if command exists
        if command_lst[0] == "cd" or command_lst[0] in ubuntu_commands:
            return True
        
        out = subprocess.run(command_lst, capture_output=True, text=True)

        # print("\n\n\nOutput:", out.stdout)  # Standard output
        # print("Error:", out.stderr)  # Standard error (if any)
        # print("Return Code:", out.returncode, "\n\n\n")  # Return code of the command

        # subprocess.check_output(f"which {command.split()[0]}", shell=True, text=True)
    except subprocess.CalledProcessError:
        # print(f"Dependency missing for: {command.split()[0]}")
        # print(f"Installing {command.split()[0]}...")
        subprocess.run(f"brew install {command.split()[0]}", shell=True, check=True)
    
    except Exception as e:
        print(f"unexpected error occured: {e}\n\n")

# Step 5: Execute Shell Command
def execute_command(state: AgentState) -> AgentState:
    shell_command = state.shell_command
    
    # Remove the "Shell Command: " prefix
    shell_command = shell_command.replace("Shell Command: ", "")
    
    # Handle unsupported commands
    if "Unsupported Command" in shell_command:
        state.execution_result = "Command not supported or cannot be executed."
        return state

    # Check dependencies and install if missing
    install_missing_packages(shell_command)

    try:
        print(f"Executing Command: {shell_command}")

        # Split by ';' and strip spaces around each command
        commands = [shell_command]
        if ";" in shell_command:
            commands = [cmd.strip() for cmd in shell_command.split(";") if cmd.strip()]
        if "&&" in shell_command:
            commands = [cmd.strip() for cmd in shell_command.split("&&") if cmd.strip()]

        all_results = []
        
        for cmd in commands:
            print(f"Executing Sub-Command: {cmd}")
            
            # Preprocess the shell command
            cmd = os.path.expanduser(cmd)
            cmd = os.path.expandvars(cmd)

            # Split command if not using `shell=True`
            shell_command_list = cmd.split()
            shell_command_list[-1] = os.path.expanduser(shell_command_list[-1])
            print(f"Processed Sub-Command: {shell_command_list}")

            # Execute the command
            result = subprocess.run(shell_command_list, capture_output=True, shell=False)
            print(result)

            # Handle errors and collect results
            if result.returncode != 0:
                error_msg = f"Error executing command '{cmd}': {result.stderr.strip()}"
                print(error_msg)
                all_results.append(error_msg)
            else:
                output = result.stdout
                print(f"Sub-Command Output:\n{output}")
                all_results.append(output)
            state.execution_result = all_results
        # Combine all results into a single output
        # state.execution_result = "\n".join(all_results)
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
        template="Explain the output in complete layman terms, and be straight to the point. : Output: {result}"
    )
    chain = prompt_template | llm
    result = chain.invoke({"result": state.execution_result})
    state.response = result.content.strip()
    # print(f"Explanation: {state.response}")
    return state

# Create LangGraph Workflow
def create_agent_graph():
    graph = Graph()
    graph.add_node("classify_input", classify_input)
    graph.add_node("handle_chat", handle_chat)
    graph.add_node("interpret_command", interpret_command)
    # NEW NODE ADDED
    graph.add_node("check_network_command",check_network_command)
    graph.add_node("execute_command", execute_command)
    graph.add_node("generate_command_response", generate_command_response)
    graph.set_entry_point("classify_input")
    graph.add_conditional_edges(
        "classify_input",
        lambda state: "handle_chat" if not state.is_command else "interpret_command"
    )
    graph.add_edge("handle_chat", END)
    # graph.add_edge("interpret_command", "check_network_command") #connect interpret to check_network_command
    graph.add_edge("interpret_command", "execute_command") #connect interpret to check_network_command
    # graph.add_edge("check_network_command", "generate_command_response") #connect check_network to execute
    graph.add_edge("execute_command", "generate_command_response")
    graph.add_edge("generate_command_response", END)
    return graph

# Command Line Interface
def main():
    print("AI Command Line Agent (powered by Gemini) - Type 'exit' to quit.")
    graph = create_agent_graph()
    app = graph.compile()

    while True:
        user_input = f"The current working directory is {present_path} " + input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        state = AgentState(user_input=user_input)
        print("THIS STATE",state.execution_result)
        result = app.invoke(state)
        print("AI Agent:", result.response)

if __name__ == "__main__":
    main()
