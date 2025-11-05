"""
Hermes AI Logistics Assistant Agent
LangChain ReAct Agent with multiple tools for logistics analysis
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from typing import Dict, List
import os

from backend.tools import ALL_TOOLS, set_data_processor
from backend.data_processor import DataProcessor


class HermesAgent:
    def __init__(self, openai_api_key: str, data_path: str = "data/shipments.csv"):
        """
        Initialize Hermes AI Logistics Assistant
        
        Args:
            openai_api_key: OpenAI API key
            data_path: Path to shipments CSV file
        """
        self.openai_api_key = openai_api_key
        
        # Initialize data processor
        self.data_processor = DataProcessor(data_path)
        set_data_processor(self.data_processor)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=openai_api_key
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Create agent
        self.agent_executor = self._create_agent()
        
        print("✓ Hermes AI Logistics Assistant initialized successfully")
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent with tools"""
        
        # Define the prompt template
        template = """You are Hermes, an AI Logistics Assistant designed to help operations managers analyze shipment data and answer logistics questions.

You have access to the following tools to query and analyze shipment data:

{tools}

IMPORTANT INSTRUCTIONS:
1. You are a logistics expert. Always provide clear, actionable insights.
2. When users ask questions, use the appropriate tools to fetch real data from the shipment database.
3. You can use multiple tools in sequence to answer complex questions.
4. For prediction questions, first gather historical data, then use prediction tools.
5. For optimization questions, analyze performance metrics before making recommendations.
6. Always cite specific numbers and data points in your responses.
7. If a question is ambiguous, make reasonable assumptions based on logistics context.
8. Format your responses clearly with proper structure (use bullet points, numbers when appropriate).
9. When showing multiple results, organize them in a clear, readable format.

RESPONSE GUIDELINES:
- Start with a direct answer to the user's question
- Support your answer with specific data and metrics
- If using tool results, interpret and explain them in business context
- Suggest actionable insights when relevant
- Be concise but comprehensive

Use the following format:

Question: the input question you must answer
Thought: think about what you need to do and which tool(s) to use
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, formatted clearly for operations managers

Remember: You can use multiple tools in sequence! For complex questions requiring predictions or analysis:
1. First get the relevant historical data
2. Then analyze trends or patterns
3. Finally make predictions or recommendations

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        
        # Create the agent
        agent = create_react_agent(
            llm=self.llm,
            tools=ALL_TOOLS,
            prompt=prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            return_intermediate_steps=True
        )
        
        return agent_executor
    
    def chat(self, user_message: str) -> Dict:
        """
        Process user message and return response
        
        Args:
            user_message: User's question or message
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Invoke the agent
            result = self.agent_executor.invoke({"input": user_message})
            
            # Extract intermediate steps for debugging
            steps = []
            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    steps.append({
                        "tool": action.tool,
                        "tool_input": action.tool_input,
                        "observation": str(observation)[:200] + "..." if len(str(observation)) > 200 else str(observation)
                    })
            
            return {
                "response": result["output"],
                "success": True,
                "intermediate_steps": steps
            }
            
        except Exception as e:
            error_msg = f"I encountered an error while processing your request: {str(e)}"
            print(f"Agent error: {e}")
            return {
                "response": error_msg,
                "success": False,
                "error": str(e)
            }
    
    def reset_memory(self):
        """Clear conversation history"""
        self.memory.clear()
        print("✓ Conversation history cleared")
    
    def get_data_summary(self) -> Dict:
        """Get summary of loaded data"""
        return self.data_processor.get_summary_stats()


def create_hermes_agent(api_key: str, data_path: str = "data/shipments.csv") -> HermesAgent:
    """
    Factory function to create Hermes agent
    
    Args:
        api_key: OpenAI API key
        data_path: Path to shipments data
        
    Returns:
        HermesAgent instance
    """
    return HermesAgent(openai_api_key=api_key, data_path=data_path)