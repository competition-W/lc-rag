## 源代码
DEFAULT_CONTEXT_PROMPT_TEMPLATE = """
  The following is a friendly conversation between a user and an AI assistant.
  The assistant is talkative and provides lots of specific details from its context.
  If the assistant does not know the answer to a question, it truthfully says it
  does not know.

  Here are the relevant documents for the context:

  {context_str}

  Instruction: Based on the above documents, provide a detailed answer for the user question below.
  Answer "don't know" if not present in the document.
  """

## 中文
context_prompt_template = """
以下是一段用户与AI助手之间的友好对话。
该助手健谈，并会根据上下文提供大量具体细节。
如果助手不知道某个问题的答案，它会如实地说不知道。
---------------------\n
以下是与上下文相关的文档内容：
{chat_history}\n
---------------------\n\n
Instruction:\n
仅基于已提供的对话历史进行回答，请使用简洁的话语回答。\n
请在严谨的立场下,不得引入任何未提及的内容,不假设未明确陈述的信息，不虚构任何细节。\n
文档中没有代码的，请不要返回代码！\n
保持与用户提问的主题一致性,保持回答自洽self-consistent。\n
如果上述文档有助于解决用户的最新提问,为下方用户问题提供详细解答。\n
如果文档中未包含相关内容，则回答“我不知道，请您尝试完善提问”。\n
"""

#########################################
#### 问题压缩
condense_prompt_template = """
你是一个智能助手，负责将对话历史和最新的用户消息压缩成一个独立的问题\n。
不能引入未提及的话题，保持上下文一致性。\n
---------------------\n
对话历史\n：
{chat_history}
新的用户消息\n：
{question}
---------------------\n
请将该后续问题重新表述为一个简洁、明确的独立问题\n。
新的独立问题：
"""

DEFAULT_CONDENSE_PROMPT_TEMPLATE = """
  Given the following conversation between a user and an AI assistant and a follow up question from user,
  rephrase the follow up question to be a standalone question.

  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:"""

################################################
DEFAULT_CONTEXT_REFINE_PROMPT_TEMPLATE = """
  The following is a friendly conversation between a user and an AI assistant.
  The assistant is talkative and provides lots of specific details from its context.
  If the assistant does not know the answer to a question, it truthfully says it
  does not know.

  Here are the relevant documents for the context:

  {context_msg}

  Existing Answer:
  {existing_answer}

  Instruction: Refine the existing answer using the provided context to assist the user.
  If the context isn't helpful, just repeat the existing answer and nothing more.
  """
###### 控制上下文
context_refine_prompt_template = """
以下是一段用户与AI助手之间的友好对话。
该助手健谈，并会根据上下文提供大量具体细节。
如果助手不知道某个问题的答案，它会如实地说不知道。
---------------------\n
以下是与上下文相关的文档内容：
对话历史：
{context_msg}
现有回答：
{existing_answer}
---------------------\n
Instruction：请使用提供的上下文对现有回答进行改进，以更好地帮助用户。
如果上下文没有帮助，请直接重复现有回答，不做任何改动。
"""