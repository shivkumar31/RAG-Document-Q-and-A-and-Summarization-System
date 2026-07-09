from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config import llm


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ask_question(vectorstore, question: str):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Answer ONLY from the given context.

If the answer is not available, reply:
'I couldn't find the answer in the document.'

Always be concise.
""",
            ),
            (
                "human",
                """Context:

{context}

Question:

{question}
""",
            ),
        ]
    )

    rag_chain = (
        {
            "context": itemgetter("question")
            | retriever
            | RunnableLambda(_format_docs),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(
        {
            "question": question
        }
    )

    docs = retriever.invoke(question)

    sources = [
        {
            "page": doc.metadata.get("page", 0) + 1
        }
        for doc in docs
    ]

    return {
        "answer": answer,
        "sources": sources,
    }


def summarize_document(chunks):
    map_prompt = ChatPromptTemplate.from_template(
        """
Summarize the following chunk.

Chunk:
{chunk}
"""
    )

    map_chain = (
        map_prompt
        | llm
        | StrOutputParser()
    )

    chunk_summaries = [
        map_chain.invoke(
            {
                "chunk": chunk.page_content
            }
        )
        for chunk in chunks
    ]

    reduce_prompt = ChatPromptTemplate.from_template(
        """
Below are summaries of every chunk of a document.

Create one well-structured final summary.

Summaries:

{summaries}
"""
    )

    reduce_chain = (
        reduce_prompt
        | llm
        | StrOutputParser()
    )

    final_summary = reduce_chain.invoke(
        {
            "summaries": "\n\n".join(chunk_summaries)
        }
    )

    return final_summary