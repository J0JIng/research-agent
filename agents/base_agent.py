from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain.agents import create_tool_calling_agent, AgentExecutor

from agents.responseFormat import basic_parser

class BaseAgent:
    def __init__(self, tools, model: str = "gpt-4o-mini"):
        self.tools = tools
        self.llm = ChatOpenAI(model=model)

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a polite and helpful assistant agent.
                Answer the user query and use necessary tools.
                Wrap the output in this format and provide no other text:
                {format_instructions}
                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]).partial(format_instructions=basic_parser.get_format_instructions())

        self.agent = create_tool_calling_agent(
            llm=self.llm,
            prompt=self.prompt,
            tools=self.tools,
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
        )

    def execute(self, query: str) -> str:
        with get_openai_callback() as cb:
            raw_response = self.agent_executor.invoke({"query": query})
            try:
                structured_response = basic_parser.parse(raw_response.get("output"))
                print(cb)
                return structured_response
            except Exception as e:
                print(cb)
                return f"Error parsing response: {e}\nRaw Response: {raw_response}"
