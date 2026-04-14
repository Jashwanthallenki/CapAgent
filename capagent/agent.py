from autogen.agentchat import ConversableAgent, Agent
from autogen.runtime_logging import log_new_agent, logging_enabled

from typing import Callable, Dict, List, Literal, Optional, Union

from capagent.indexing import load_vector_store, query_vector_store

def checks_terminate_message(msg):
    if isinstance(msg, str):
        return msg.find("TERMINATE") > -1
    elif isinstance(msg, dict) and 'content' in msg:
        return msg['content'].find("TERMINATE") > -1
    else:
        print(type(msg), msg)
        raise NotImplementedError
    

class CustomUserProxyAgent(ConversableAgent):
    """(In preview) A proxy agent for the user, that can execute code and provide feedback to the other agents.

    UserProxyAgent is a subclass of ConversableAgent configured with `human_input_mode` to ALWAYS
    and `llm_config` to False. By default, the agent will prompt for human input every time a message is received.
    Code execution is enabled by default. LLM-based auto reply is disabled by default.
    To modify auto reply, register a method with [`register_reply`](conversable_agent#register_reply).
    To modify the way to get human input, override `get_human_input` method.
    To modify the way to execute code blocks, single code block, or function call, override `execute_code_blocks`,
    `run_code`, and `execute_function` methods respectively.
    """

    # Default UserProxyAgent.description values, based on human_input_mode
    DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS = {
        "ALWAYS": "An attentive HUMAN user who can answer questions about the task, and can perform tasks such as running Python code or inputting command line commands at a Linux terminal and reporting back the execution results.",
        "TERMINATE": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
        "NEVER": "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks).",
    }

    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "TERMINATE", "NEVER"] = "ALWAYS",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Union[Dict, Literal[False]] = {},
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        llm_config: Optional[Union[Dict, Literal[False]]] = False,
        system_message: Optional[Union[str, List]] = "",
        description: Optional[str] = None,
    ):
        """
        Args:
            name (str): name of the agent.
            is_termination_msg (function): a function that takes a message in the form of a dictionary
                and returns a boolean value indicating if this received message is a termination message.
                The dict can contain the following keys: "content", "role", "name", "function_call".
            max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
                default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
                The limit only plays a role when human_input_mode is not "ALWAYS".
            human_input_mode (str): whether to ask for human inputs every time a message is received.
                Possible values are "ALWAYS", "TERMINATE", "NEVER".
                (1) When "ALWAYS", the agent prompts for human input every time a message is received.
                    Under this mode, the conversation stops when the human input is "exit",
                    or when is_termination_msg is True and there is no human input.
                (2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or
                    the number of auto reply reaches the max_consecutive_auto_reply.
                (3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops
                    when the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.
            function_map (dict[str, callable]): Mapping function names (passed to openai) to callable functions.
            code_execution_config (dict or False): config for the code execution.
                To disable code execution, set to False. Otherwise, set to a dictionary with the following keys:
                - work_dir (Optional, str): The working directory for the code execution.
                    If None, a default working directory will be used.
                    The default working directory is the "extensions" directory under
                    "path_to_autogen".
                - use_docker (Optional, list, str or bool): The docker image to use for code execution.
                    Default is True, which means the code will be executed in a docker container. A default list of images will be used.
                    If a list or a str of image name(s) is provided, the code will be executed in a docker container
                    with the first image successfully pulled.
                    If False, the code will be executed in the current environment.
                    We strongly recommend using docker for code execution.
                - timeout (Optional, int): The maximum execution time in seconds.
                - last_n_messages (Experimental, Optional, int): The number of messages to look back for code execution. Default to 1.
            default_auto_reply (str or dict or None): the default auto reply message when no code execution or llm based reply is generated.
            llm_config (dict or False or None): llm inference configuration.
                Please refer to [OpenAIWrapper.create](/docs/reference/oai/client#create)
                for available options.
                Default to False, which disables llm-based auto reply.
                When set to None, will use self.DEFAULT_CONFIG, which defaults to False.
            system_message (str or List): system message for ChatCompletion inference.
                Only used when llm_config is not False. Use it to reprogram the agent.
            description (str): a short description of the agent. This description is used by other agents
                (e.g. the GroupChatManager) to decide when to call upon this agent. (Default: system_message)
        """
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=(
                description if description is not None else self.DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS[human_input_mode]
            ),
        )

        if logging_enabled():
            log_new_agent(self, locals())



class CapAgent(CustomUserProxyAgent):
    
    def __init__(
        self,
        name,
        prompt_generator, 
        parser,
        executor,
        **config,
    ):
        super().__init__(name=name, **config)
        self.prompt_generator = prompt_generator
        self.parser = parser
        self.executor = executor
        
    def sender_hits_max_reply(self, sender: Agent):
        return self._consecutive_auto_reply_counter[sender.name] >= self._max_consecutive_auto_reply

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        """Receive a message from the sender agent.
        Once a message is received, this function sends a reply to the sender or simply stop.
        The reply can be generated automatically or entered manually by a human.
        """
        
        print("COUNTER:", self._consecutive_auto_reply_counter[sender.name])

        self._process_received_message(message, sender, silent)
        
        # parsing the code component, if there is one
        parsed_results = self.parser.parse(message)
        parsed_content = parsed_results['content']
        parsed_status = parsed_results['status']
        parsed_error_message = parsed_results['message']
        parsed_error_code = parsed_results['error_code']
        
        # if TERMINATION message, then return
        if not parsed_status and self._is_termination_msg(message):
            return
        
        # if parsing fails
        if not parsed_status:
            
            # reset the consecutive_auto_reply_counter
            if self.sender_hits_max_reply(sender):
                self._consecutive_auto_reply_counter[sender.name] = 0
                return
            
            # if parsing fails, construct a feedback message from the error code and message of the parser
            # send the feedback message, and request a reply
            self._consecutive_auto_reply_counter[sender.name] += 1
            reply = self.prompt_generator.get_parsing_feedback(parsed_error_message, parsed_error_code)
            self.feedback_types.append("parsing")
            self.send(reply, sender, request_reply=True)
            return
        
        # if parsing succeeds, then execute the code component
        if self.executor:
            # go to execution stage if there is an executor module
            exit_code, output, file_paths = self.executor.execute(parsed_content)
            reply = self.prompt_generator.get_exec_feedback(exit_code, output)
            
            # if execution fails
            if exit_code != 0:
                if self.sender_hits_max_reply(sender):
                    # reset the consecutive_auto_reply_counter
                    self._consecutive_auto_reply_counter[sender.name] = 0
                    return
                
                self._consecutive_auto_reply_counter[sender.name] += 1
                self.send(reply, sender, request_reply=True)
                return
                
            # if execution succeeds
            else:
                self.send(reply, sender, request_reply=True)
                self._consecutive_auto_reply_counter[sender.name] = 0
                return
    
    def generate_init_message(self, query, n_image, cot_examples):  
        content = self.prompt_generator.initial_prompt(query, n_image, cot_examples)
        return content
    
    def get_cot_examples(self, query_str: str):
        vector_store = load_vector_store("cot_examples")
        query_result = query_vector_store(vector_store, query_str, "default", similarity_top_k=2)
        cot_examples = "\n".join([node.text for node in query_result.nodes])
        return cot_examples

    def initiate_chat(self, assistant, message, n_image=0, log_prompt_only=False, use_rag=True):

        self.feedback_types = []
        
        if use_rag:
            print("Using RAG to get CoT examples ...")
            print(f"Query string: {message}")
            cot_examples = self.get_cot_examples(message)
            print(f"Retrieved CoT examples: \n{cot_examples}")
        else:
            cot_examples = ""
        
        initial_message = self.generate_init_message(message, n_image, cot_examples)
        
        if log_prompt_only:
            print(initial_message)
        else:
            assistant.receive(initial_message, self, request_reply=True)

        chain_of_thought = self.get_chain_of_thought(assistant)
        result = self.result_parser(self._oai_messages[assistant][-1]['content'])
        return result, chain_of_thought
    
    def result_parser(self, result):
        result = result.split("ANSWER:")[1].replace("TERMINATE", "").strip()
        return result
    
    def get_chain_of_thought(self, assistant):
        messages = []
        for message in self._oai_messages[assistant]:
            if message['name'] == assistant.name:
                messages.append({'role': 'assistant', 'content': message['content']})
            elif message['content'].startswith("OBSERVATION"):
                messages.append({'role': 'user', 'content': message['content']})
        return messages
    
    