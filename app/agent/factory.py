"""Construction of the LangChain conversational clinical assistant."""
from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent

from app.agent.history import get_session_history
from app.config import settings
from app.diagnosis.service import DifferentialDiagnosis
from app.nlp.extractor import text_to_features


SYSTEM_PROMPT = """You are an AI Clinical Decision Support Assistant.
Your role is to collect symptom information conversationally and use the supplied diagnostic tool when enough information is available.

Rules:
1. Do not repeat greetings once the conversation is underway.
2. When the patient provides symptoms or relevant patient details, use the run_differential_diagnosis tool.
3. Explain that model probabilities are decision-support outputs, not confirmed diagnoses.
4. Clearly present Malaria and COVID-19 probabilities and the available next-step recommendations.
5. Do not invent test results, medical history, vital signs, or diagnoses that were not supplied or returned by the tool.
6. If urgent symptoms are described, advise prompt in-person medical assessment.
7. Properly format tool response to be user-friendly and easily understandable and to look like human written text.
"""


def create_conversational_agent(diagnosis_engine: DifferentialDiagnosis | None = None):
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured. Add it to .env before starting the conversational agent.")

    engine = diagnosis_engine or DifferentialDiagnosis()

    @tool
    def run_differential_diagnosis(patient_text_description: str) -> str:
        """Run the Malaria vs COVID-19 differential pipeline on a patient description."""
        features = text_to_features(patient_text_description)
        return json.dumps(engine.diagnose(features), default=str)

    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.2,
    )
    tools = [run_differential_diagnosis]
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return RunnableWithMessageHistory(
        executor,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="output",
    )
